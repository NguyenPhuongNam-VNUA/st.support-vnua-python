from fastapi import APIRouter
from app.api.v1.endpoints import ask, questions, tokens, auth

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(ask.router, tags=["Chatbot RAG"])
api_router.include_router(questions.router, tags=["Questions & Embeddings"])
api_router.include_router(tokens.router, tags=["Tokens & Utils"])
