import asyncio
from aiohttp import web
from config.logger import setup_logging
from core.api.ota_handler import OTAHandler
from core.api.vision_handler import VisionHandler

TAG = __name__


class SimpleHttpServer:
    def __init__(self, config: dict, ws_server=None):
        self.config = config
        self.logger = setup_logging()
        self.ota_handler = OTAHandler(config)
        self.vision_handler = VisionHandler(config)
        self.ws_server = ws_server  # 调试端点用：读取活跃连接的视频缓冲帧

    async def handle_debug_video_frames(self, request):
        """调试端点：把活跃会话视频缓冲中的帧 dump 成 JPEG 文件，返回保存路径列表。

        GET /debug/video_frames?count=5  （count 默认 5，最多 20）
        仅用于本地调试（无鉴权，请勿在公网暴露）。"""
        import os
        import time as _time

        from aiohttp import web as _web

        if self.ws_server is None:
            return _web.json_response({"error": "ws_server 未注入"}, status=503)
        count = int(request.query.get("count", "5"))
        count = max(1, min(count, 20))

        dumped = []
        debug_info = []
        out_dir = os.path.join("data", "video", "debug_dump")
        os.makedirs(out_dir, exist_ok=True)
        for handler in list(getattr(self.ws_server, "active_handlers", [])):
            session = getattr(handler, "video_session", None)
            frame_count = 0
            if session is not None:
                frames = session._frames.snapshot() if hasattr(
                    session, "_frames"
                ) else []
                frame_count = len(frames)
            debug_info.append(
                {
                    "session": handler.session_id,
                    "has_video_session": session is not None,
                    "buffered_frames": frame_count,
                }
            )
            if session is None:
                continue
            frames = session._frames.snapshot() if hasattr(
                session, "_frames"
            ) else []
            if not frames:
                continue
            step = max(1, len(frames) // count)
            picked = frames[::step][:count]
            for idx, frame in enumerate(picked):
                name = (
                    f"{handler.session_id[:8]}_{int(_time.time())}_{idx:02d}.jpg"
                )
                path = os.path.join(out_dir, name)
                with open(path, "wb") as fh:
                    fh.write(frame.data)
                dumped.append({"session": handler.session_id, "file": path})
        return _web.json_response({"dumped": dumped, "handlers": debug_info})

    def _get_websocket_url(self, local_ip: str, port: int) -> str:
        """获取websocket地址

        Args:
            local_ip: 本地IP地址
            port: 端口号

        Returns:
            str: websocket地址
        """
        server_config = self.config["server"]
        websocket_config = server_config.get("websocket")

        if websocket_config and "你" not in websocket_config:
            return websocket_config
        else:
            return f"ws://{local_ip}:{port}/xiaozhi/v1/"

    async def start(self):
        try:
            server_config = self.config["server"]
            read_config_from_api = self.config.get("read_config_from_api", False)
            host = server_config.get("ip", "0.0.0.0")
            port = int(server_config.get("http_port", 8003))

            if port:
                app = web.Application()

                if not read_config_from_api:
                    # 如果没有开启智控台，只是单模块运行，就需要再添加简单OTA接口，用于下发websocket接口
                    app.add_routes(
                        [
                            web.get("/xiaozhi/ota/", self.ota_handler.handle_get),
                            web.post("/xiaozhi/ota/", self.ota_handler.handle_post),
                            web.options(
                                "/xiaozhi/ota/", self.ota_handler.handle_options
                            ),
                            # 下载接口，仅提供 data/bin/*.bin 下载
                            web.get(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_download,
                            ),
                            web.options(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_options,
                            ),
                        ]
                    )
                # 添加路由
                app.add_routes(
                    [
                        web.get(
                            "/debug/video_frames",
                            self.handle_debug_video_frames,
                        ),
                        web.get("/mcp/vision/explain", self.vision_handler.handle_get),
                        web.post(
                            "/mcp/vision/explain", self.vision_handler.handle_post
                        ),
                        web.options(
                            "/mcp/vision/explain", self.vision_handler.handle_options
                        ),
                    ]
                )

                # 运行服务
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, host, port)
                await site.start()

                # 保持服务运行
                while True:
                    await asyncio.sleep(3600)  # 每隔 1 小时检查一次
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"HTTP服务器启动失败: {e}")
            import traceback

            self.logger.bind(tag=TAG).error(f"错误堆栈: {traceback.format_exc()}")
            raise
