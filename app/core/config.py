import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "ST-Care"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"

    # API Keys & URLs
    GOOGLE_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/st_care_db"
    LARAVEL_API_BASE_URL: str = "http://127.0.0.1:8000/api"
    PUBLIC_QUESTION_SECRET: str = ""
    PORT: int = 5001

    # JWT Authentication
    JWT_SECRET_KEY: str = "st_support_vnua_super_secret_jwt_key_2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "https://st-dse.vnua.edu.vn:6896",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "*"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton Pattern
settings = Settings()
