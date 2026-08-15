from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str
    response_type: str  # 'answered' | 'not_found' | 'out_of_topic' | 'auto_generated'
    context: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
