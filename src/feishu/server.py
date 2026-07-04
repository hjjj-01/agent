"""
飞书机器人Web服务器

功能：
1. 接收飞书推送的事件
2. 调用消息处理器处理事件
3. 返回响应给飞书

使用FastAPI框架创建Web服务器。
飞书通过Webhook将事件推送到这个服务器。

飞书事件推送流程：
飞书用户发送消息 → 飞书服务器 → HTTP POST请求到我们的服务器 → 
处理事件 → 返回HTTP响应 → 飞书服务器确认收到

重要：飞书要求 webhook 在 3 秒内返回响应！如果 Agent 同步处理耗时
超过 3 秒，飞书会认为超时并重试，导致同一条消息被处理多次。
因此这里使用 BackgroundTasks 异步处理：收到事件后立刻返回 200，
然后在后台线程中调用 Agent 并发送回复。
"""
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn
import json
import asyncio

# 导入飞书机器人
from .bot import FeishuBot, MessageHandler
from ..agent.wms_agent import WMSAgent
from ..utils.config import Config


class FeishuEvent(BaseModel):
    """
    飞书事件数据模型

    飞书推送的事件数据结构。
    """
    schema: str
    header: Dict[str, Any]
    event: Dict[str, Any]


class FeishuServer:
    """
    飞书机器人Web服务器

    使用FastAPI创建Web服务器，接收和处理飞书事件。

    使用方式：
        server = FeishuServer(bot, port=8080)
        server.start()
    """

    def __init__(self, bot: FeishuBot, port: int = 8080):
        """
        初始化飞书服务器

        Args:
            bot: 飞书机器人实例
            port: 服务端口
        """
        logger.info(f"初始化飞书Web服务器，端口: {port}")

        self.bot = bot
        self.port = port
        self.message_handler = MessageHandler(bot)

        # 创建FastAPI应用
        self.app = FastAPI(title="WMS AI Agent - 飞书机器人")

        # 注册路由
        self._setup_routes()

        logger.success("飞书Web服务器初始化完成")

    def _setup_routes(self):
        """
        设置FastAPI路由

        路由说明：
            - /webhook: 接收飞书事件推送的主路由
            - /health: 健康检查路由（用于监控）
            - /test: 测试路由（用于开发调试）
        """
        logger.info("设置FastAPI路由")

        @self.app.post("/webhook")
        async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
            """
            处理飞书Webhook请求

            飞书将事件推送到这个路由。
            
            关键设计：3秒超时问题
            ============================================================
            飞书要求 webhook 在 3 秒内返回 HTTP 200，否则认为超时并重试。
            但 Agent 调用 LLM 可能需要 5-10 秒，所以不能同步等待。
            
            解决方案：
            1. 收到事件 → 立刻返回 200（飞书停止重试）
            2. 把事件处理放进 BackgroundTasks（后台异步执行）
            3. 后台线程慢慢调 Agent、发回复
            ============================================================
            """
            logger.info("收到飞书Webhook请求")

            try:
                # 获取请求体
                body = await request.json()

                logger.info(f"请求体: {json.dumps(body, ensure_ascii=False)[:200]}...")

                # ============================================================
                # URL 验证（飞书配置事件订阅地址时触发）
                # ============================================================
                if body.get("type") == "url_verification":
                    challenge = body.get("challenge", "")
                    logger.info(f"URL 验证请求，返回 challenge: {challenge}")
                    return JSONResponse(
                        status_code=200,
                        content={"challenge": challenge}
                    )

                # 解析事件类型
                header = body.get("header", {})
                event_type = header.get("event_type", "")

                # ============================================================
                # 异步处理：把耗时操作放进后台任务
                # ============================================================
                # 飞书可能重试，所以靠 event_id 去重可以避免重复处理
                # 这里先用 background_tasks 解决超时问题，去重后面再优化
                # ============================================================
                if event_type == "im.message.receive_v1":
                    # 放入后台任务，立刻返回 200
                    # add_task 会在后台线程中执行，不阻塞 webhook 响应
                    background_tasks.add_task(
                        self._process_event_async,
                        event_type,
                        body
                    )
                else:
                    # 非消息事件（如审批、回调等），同步处理即可
                    self.message_handler.handle_event(event_type, body)

                # 立刻返回 200，飞书不会重试
                return JSONResponse(
                    status_code=200,
                    content={"code": 0, "msg": "success"}
                )

            except Exception as e:
                logger.error(f"处理Webhook请求失败: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"code": 500, "msg": str(e)}
                )

        @self.app.get("/health")
        async def health_check():
            """
            健康检查接口

            用于监控服务器状态。
            """
            return {"status": "healthy", "service": "WMS AI Agent"}

        @self.app.get("/")
        async def root():
            """
            根路由

            显示服务信息。
            """
            return {
                "service": "WMS AI Agent",
                "version": "1.0",
                "description": "WMS仓库管理智能助手 - 飞书机器人版本"
            }

        @self.app.post("/test")
        async def test_chat(request: Request):
            """
            测试聊天接口

            用于开发调试，可以直接向Agent发送消息。
            不经过飞书，直接调用Agent。

            请求格式：
            {
                "message": "查询SKU001的库存"
            }
            """
            logger.info("收到测试聊天请求")

            try:
                body = await request.json()
                user_message = body.get("message", "")

                if not user_message:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "缺少message参数"}
                    )

                # 直接调用Agent
                response = self.bot.agent.chat(user_message)

                return JSONResponse(
                    status_code=200,
                    content={
                        "user_message": user_message,
                        "agent_response": response,
                        "tools_available": self.bot.agent.get_available_tools()
                    }
                )

            except Exception as e:
                logger.error(f"测试聊天失败: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )

    def _process_event_async(self, event_type: str, event_data: Dict[str, Any]):
        """
        后台异步处理飞书事件

        这个方法由 BackgroundTasks 在后台线程中调用，
        不会阻塞 webhook 的 HTTP 响应，从而避免飞书 3 秒超时重试。

        Args:
            event_type: 事件类型（如 im.message.receive_v1）
            event_data: 完整的事件数据
        """
        try:
            logger.info(f"后台处理事件: {event_type}")
            result = self.message_handler.handle_event(event_type, event_data)
            logger.info(f"后台处理完成: {result}")
        except Exception as e:
            logger.error(f"后台处理事件失败: {str(e)}")

    def start(self):
        """
        启动Web服务器

        使用uvicorn运行FastAPI应用。

        说明：
            uvicorn是一个高性能的ASGI服务器。
            飞书要求服务器能够响应HTTP请求。
        """
        logger.info(f"启动飞书Web服务器，端口: {self.port}")

        # 运行服务器
        uvicorn.run(
            self.app,
            host="0.0.0.0",  # 监听所有网络接口
            port=self.port,
            log_level="info"
        )


def create_feishu_server(config: Config, agent: WMSAgent) -> FeishuServer:
    """
    创建飞书服务器

    Args:
        config: 配置对象
        agent: WMS Agent实例

    Returns:
        FeishuServer实例

    说明：
        这是创建飞书服务器的便捷函数。
    """
    # 创建飞书机器人
    bot = FeishuBot(
        app_id=config.feishu.app_id,
        app_secret=config.feishu.app_secret,
        verification_token=config.feishu.verification_token,
        agent=agent,
        encrypt_key=config.feishu.encrypt_key
    )

    # 创建服务器
    server = FeishuServer(bot, port=config.server.port)

    return server