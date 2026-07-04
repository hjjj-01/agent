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
    embedding_model: str = Field(default="text-embedding-3-small", description="embedding模型名称")
    embedding_api_base: str = Field(default=None, description="embedding API基础URL（None表示使用api_base）")
    aliyun_api_key: str = Field(default=None, description="阿里云API密钥（用于调用DashScope embedding API）")


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
    port: int = Field(default=5000, description="服务端口")
    log_level: str = Field(default="INFO", description="日志级别")


class Config(BaseModel):
    """应用总配置"""
    openai: OpenAIConfig
    wms: WMSConfig
    feishu: FeishuConfig
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

# 调用函数获取所有的配置
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
    # 加载.env文件到环境变量
    load_dotenv(env_file)

    # 从环境变量读取并构建对象
    config = Config(
        openai=OpenAIConfig(
            #os.getenv("变量名", "默认值") 读取环境变量
            api_key=os.getenv("OPENAI_API_KEY", ""),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_base=os.getenv("OPENAI_EMBEDDING_API_BASE"),
            aliyun_api_key=os.getenv("ALIYUN_API_KEY")
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
            port=int(os.getenv("SERVER_PORT", "5000")),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    )

    return config