"""
ChatBot 演示应用配置

此配置独立于 HiveMemory 核心系统配置，专门用于 ChatBot 演示应用。
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from hivememory.patchouli.config import LLMConfig

class ChatBotConfig(BaseSettings):
    """
    ChatBot 应用配置
    
    加载优先级:
    1. 环境变量 (CHATBOT__*)
    2. .env 文件
    3. 默认值
    """
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4o"))
    
    model_config = SettingsConfigDict(
        env_file=(".env", "configs/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix="CHATBOT__"
    )

def load_chatbot_config() -> ChatBotConfig:
    """加载 ChatBot 配置"""
    return ChatBotConfig()
