from sqlmodel import SQLModel, Field, Column
from pgvector.sqlalchemy import Vector
from typing import Optional, List

class Question(SQLModel, table=True):
    __tablename__ = "questions"

    id: Optional[int] = Field(default=None, primary_key=True)
    question: str = Field(index=True)
    answer: Optional[str] = Field(default=None)
    topic: Optional[str] = Field(default="Chưa phân loại")
    related_questions: Optional[str] = Field(default=None)
    has_answer: bool = Field(default=True)
    ask_count: int = Field(default=1)
    is_embed: bool = Field(default=False)

    # Cột vector 1536 chiều từ Gemini Embeddings (pgvector)
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(1536)))
