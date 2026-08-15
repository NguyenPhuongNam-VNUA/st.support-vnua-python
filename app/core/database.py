from sqlmodel import create_engine, Session
from app.core.config import settings

# Tạo Engine kết nối PostgreSQL bằng SQLAlchemy / SQLModel
engine = create_engine(settings.DATABASE_URL, echo=False)

def get_session():
    """Dependency cung cấp Database Session cho các route FastAPI"""
    with Session(engine) as session:
        yield session
