"""
DocMind 统一配置管理
支持从环境变量和 .env 文件加载配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class Config:
    """应用配置类"""

    # ===== DeepSeek API =====
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ===== 嵌入模型 =====
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "mps")  # Apple Silicon GPU

    # ===== Redis =====
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # ===== ChromaDB =====
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")

    # ===== 文档处理 =====
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

    # ===== 服务 =====
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> bool:
        """验证必要的配置项"""
        if not cls.DEEPSEEK_API_KEY or cls.DEEPSEEK_API_KEY == "sk-your-api-key-here":
            print("⚠️  警告: 未设置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入你的 API Key")
            print("   获取 API Key: https://platform.deepseek.com/api_keys")
            return False
        return True


config = Config()
