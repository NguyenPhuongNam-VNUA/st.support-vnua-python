from pydantic import BaseModel
from typing import List, Optional, Any

class CheckDuplicateRequest(BaseModel):
    question: str
    related_questions: Optional[str] = None

class EmbedRequest(BaseModel):
    id: Any
    question: str
    answer: Optional[str] = ""
    has_answer: Optional[bool] = True
    topic: Optional[str] = ""
    related_questions: Optional[str] = ""

class CountTokenRequest(BaseModel):
    text: str

class QuestionItem(BaseModel):
    id: Any
    question: str
    answer: Optional[str] = ""
    has_answer: Optional[bool] = True
    topic: Optional[str] = ""
    related_questions: Optional[str] = ""

class EmbedBatchRequest(BaseModel):
    questions: List[QuestionItem]

class DeleteEmbedRequest(BaseModel):
    id: Any

class DeleteEmbedManyRequest(BaseModel):
    ids: List[Any]

class AskRequest(BaseModel):
    question: str
    messages: Optional[List[dict]] = []
