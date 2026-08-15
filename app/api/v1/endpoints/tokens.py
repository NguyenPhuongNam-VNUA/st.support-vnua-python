from fastapi import APIRouter
from app.schemas.question import CountTokenRequest
from app.services.embedding_service import CustomGeminiEmbeddings

router = APIRouter()

@router.post("/countToken")
async def count_input_tokens(payload: CountTokenRequest):
    tokenize = CustomGeminiEmbeddings.get_input_token_count(payload.text)
    return {"token_count": tokenize}
