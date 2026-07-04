"""
飞书机器人主模块

功能：
1. 连接飞书开放平台
2. 接收飞书消息事件
3. 发送消息回复
4. 与WMS Agent整合

飞书机器人工作流程：
飞书用户发送消息 → 飞书服务器 → Web服务器接收事件 → 
调用WMS Agent处理 → 生成回答 → 发送回复消息 → 飞书用户收到回复

飞书开放平台配置：
1. 创建飞书应用（获取app_id和app_secret）
2. 配置事件订阅URL（指向我们的Web服务器）
3. 启用消息接收权限
4. 配置机器人能力
"""
from typing import Optional, Dict, Any
from loguru import logger
import json

# 飞书SDK
import lark_oapi as lark
from lark_oapi.api.im.v1 import *

# 导入Agent
from ..agent.wms_agent import WMSAgent


class FeishuBot:
    """
    飞书机器人类

    使用飞书开放平台SDK实现机器人功能。

    使用方式：
        bot = FeishuBot(app_id, app_secret, agent)
        bot.start()
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        verification_token: str,
        agent: WMSAgent,
        encrypt_key: Optional[str] = None
    ):
        """
        初始化飞书机器人

        Args:
            app_id: 飞书应用ID
            app_secret: 飞书应用密钥
            verification_token: 验证token（用于验证事件来源）
            agent: WMS Agent实例
            encrypt_key: 事件加密密钥（可选）

        说明：
            app_id和app_secret从飞书开放平台获取：
            1. 登录飞书开放平台：https://open.feishu.cn
            2. 创建应用或使用已有应用
            3. 在应用详情页获取凭证信息
        """
        logger.info("初始化飞书机器人")

        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.agent = agent

        # 创建飞书客户端
        # 飞书客户端用于调用飞书API，如发送消息等
        self.client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.ERROR) \
            .build()

        logger.success("飞书机器人初始化完成")

    def handle_message_event(self, event_data: Dict[str, Any]) -> str:
        """
        处理飞书消息事件

        Args:
            event_data: 飞书事件数据

        Returns:
            处理结果

        说明：
            当飞书用户发送消息给机器人时，飞书会推送事件到我们的服务器。
            这个方法处理这个事件，调用Agent生成回答，并发送回复。

        事件数据结构：
            {
                "event": {
                    "sender": {
                        "sender_id": { "open_id": "ou_xxxx" }
                    },
                    "message": {
                        "message_id": "om_xxxx",
                        "content": "{\"text\":\"查询库存\"}"
                    },
                    "chat_id": "oc_xxxx"
                }
            }
        """
        logger.info("处理飞书消息事件")

        try:
            # 解析事件数据
            # 飞书事件 v2.0 格式：
            # { "schema": "2.0", "header": {...}, "event": {
            #     "message": { "chat_id": "oc_xxx", "content": "...", ... },
            #     "sender": { "sender_id": { "open_id": "ou_xxx" } }
            # }}
            event = event_data.get("event", {})
            message = event.get("message", {})
            sender = event.get("sender", {})
            # chat_id 在 message 对象里，不是 event 的直接子字段！
            chat_id = message.get("chat_id", "")

            # 获取发送者ID
            sender_id = sender.get("sender_id", {}).get("open_id", "")

            # 解析消息内容
            # 飞书消息内容是JSON字符串，需要解析
            content_str = message.get("content", "{}")
            content_data = json.loads(content_str)

            # 提取文本内容
            # 不同消息类型有不同的content结构
            # 文本消息：{"text": "消息内容"}
            user_message = content_data.get("text", "")

            if not user_message:
                logger.warning("消息内容为空")
                return "消息内容为空"

            logger.info(f"收到用户消息: {user_message}")
            logger.info(f"发送者: {sender_id}, 聊天ID: {chat_id}")

            # 调用Agent处理消息
            # Agent会自动分析问题、调用工具、生成回答
            agent_response = self.agent.chat(user_message)

            logger.info(f"Agent回答: {agent_response[:100]}...")

            # 发送回复消息给用户
            self.send_message(chat_id, agent_response, message.get("message_id", ""))

            return "消息处理成功"

        except Exception as e:
            logger.error(f"处理飞书消息事件失败: {str(e)}")
            # 发送错误提示
            try:
                self.send_message(
                    chat_id,
                    f"抱歉，处理您的消息时出现错误：{str(e)}",
                    message.get("message_id", "")
                )
            except:
                pass
            return f"处理失败：{str(e)}"

    def send_message(
        self,
        chat_id: str,
        content: str,
        reply_to_message_id: Optional[str] = None
    ) -> bool:
        """
        发送飞书消息

        Args:
            chat_id: 聊天ID
            content: 消息内容
            reply_to_message_id: 要回复的消息ID（可选）

        Returns:
            是否发送成功

        说明：
            使用飞书API发送消息。
            可以是回复消息（指定reply_to_message_id）或新消息。
        """
        logger.info(f"发送飞书消息到 {chat_id}")

        try:
            # 构建消息内容
            # 飞书文本消息的JSON格式
            message_content = json.dumps({
                "text": content
            })

            # 创建发送消息请求
            request = CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(message_content)
                    .build()
                ) \
                .build()

            # 如果是回复消息，设置reply_to
            if reply_to_message_id:
                # 回复消息需要使用不同的方法
                # 这里简化处理，直接发送新消息
                pass

            # 发送请求
            response = self.client.im.v1.message.create(request)

            # 检查响应
            if response.success():
                logger.success(f"消息发送成功，消息ID: {response.data.message_id}")
                return True
            else:
                logger.error(f"消息发送失败: {response.code} - {response.msg}")
                return False

        except Exception as e:
            logger.error(f"发送飞书消息失败: {str(e)}")
            return False

    def get_bot_info(self) -> Dict[str, Any]:
        """
        获取机器人信息

        Returns:
            机器人信息字典
        """
        try:
            # 调用飞书API获取机器人信息
            request = GetBotInfoRequest.builder().build()
            response = self.client.im.v1.botInfo.get(request)

            if response.success():
                return {
                    "bot_id": response.data.bot_id,
                    "app_id": response.data.app_id,
                    "activate_status": response.data.activate_status,
                }
            else:
                logger.error(f"获取机器人信息失败: {response.msg}")
                return {}

        except Exception as e:
            logger.error(f"获取机器人信息失败: {str(e)}")
            return {}


class MessageHandler:
    """
    消息处理器

    处理飞书Web服务器接收到的各种事件。
    """

    def __init__(self, bot: FeishuBot):
        """
        初始化消息处理器

        Args:
            bot: 飞书机器人实例
        """
        self.bot = bot
        logger.info("消息处理器初始化完成")

    def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """
        处理飞书事件

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            处理结果

        说明：
            飞书会推送多种事件类型：
            - im.message.receive_v1: 接收到消息
            - 其他事件类型...
        """
        logger.info(f"处理飞书事件: {event_type}")

        # 根据事件类型分发处理
        if event_type == "im.message.receive_v1":
            # 消息接收事件
            return self.bot.handle_message_event(event_data)
        else:
            logger.warning(f"未处理的事件类型: {event_type}")
            return "事件类型未处理"

    def verify_request(self, request_data: Dict[str, Any]) -> bool:
        """
        验证飞书请求

        Args:
            request_data: 请求数据

        Returns:
            是否验证通过

        说明：
            验证请求是否来自飞书服务器，防止伪造请求。
        """
        # 检查token
        token = request_data.get("token", "")
        if token != self.bot.verification_token:
            logger.error("请求验证失败：token不匹配")
            return False

        return True