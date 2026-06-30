"""
配置加载模块
功能：从环境变量和配置文件中加载应用配置
"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class OpenAIConfig(BaseModel):
    """OpenAI配置"""
    api_key: str = Field(..., description="OpenAI API密钥")
    api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API基础URL")
    model: str = Field(default="gpt-4-turbo-preview", description="使用的模型名称")


class WMSConfig(BaseModel):
    """WMS系统配置"""
    api_base_url: str = Field(..., description="WMS系统API地址")
    api_token: str = Field(..., description="WMS系统认证token")


class FeishuConfig(BaseModel):
    """飞书机器人配置"""
    app_id: str = Field(..., description="飞书应用ID")
    app_secret: str = Field(..., description="飞书应用密钥")
    verification_token: str = Field(..., description="飞书机器人验证token")
    encrypt_key: Optional[str] = Field(None, description="飞书事件加密密钥")


class VectorDBConfig(BaseModel):
    """向量数据库配置"""
    db_path: str = Field(default="./data/chroma_db", description="向量数据库持久化路径")
    chunk_size: int = Field(default=500, description="文档块大小")
    chunk_overlap: int = Field(default=50, description="文档块重叠大小")


class ServerConfig(BaseModel):
    """服务器配置"""
    port: int = Field(default=8080, description="服务端口")
    log_level: str = Field(default="INFO", description="日志级别")


class Config(BaseModel):
    """应用总配置"""
    openai: OpenAIConfig
    wms: WMSConfig
    feishu: FeishuConfig
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def load_config(env_file: str = ".env") -> Config:
    """
    加载配置

    Args:
        env_file: 环境变量文件路径

    Returns:
        Config对象

    说明:
        配置加载顺序：
        1. 加载.env文件
        2. 从环境变量中读取配置
        3. 验证配置有效性
    """
    # 加载.env文件
    load_dotenv(env_file)

    # 构建配置对象
    config = Config(
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        ),
        wms=WMSConfig(
            api_base_url=os.getenv("WMS_API_BASE_URL", ""),
            api_token=os.getenv("WMS_API_TOKEN", "")
        ),
        feishu=FeishuConfig(
            app_id=os.getenv("FEISHU_APP_ID", ""),
            app_secret=os.getenv("FEISHU_APP_SECRET", ""),
            verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN", ""),
            encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY")
        ),
        vector_db=VectorDBConfig(
            db_path=os.getenv("VECTOR_DB_PATH", "./data/chroma_db"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
        ),
        server=ServerConfig(
            port=int(os.getenv("SERVER_PORT", "8080")),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    )

    return config