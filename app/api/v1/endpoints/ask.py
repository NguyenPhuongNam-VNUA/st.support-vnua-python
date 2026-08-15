from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from app.schemas.question import AskRequest
from app.services.embedding_service import CustomGeminiEmbeddings
from app.services.rag_service import generate_rag_answer, build_context
from app.core.database import get_session
from app.models.question import Question
from app.models.conversation import Conversation
from sqlmodel import select

router = APIRouter()

@router.post("/ask")
async def ask_chatbot(payload: AskRequest, db: Session = Depends(get_session)):
    question = payload.question
    history = payload.messages or []

    # 1. Embedding câu hỏi
    embedding_service = CustomGeminiEmbeddings(task_type="RETRIEVAL_QUERY")
    vector_query = embedding_service.embed_query(question)

    # 2. Query từ PostgreSQL với pgvector (Top 1)
    score_expr = (1 - Question.embedding.cosine_distance(vector_query)).label("score")
    statement = (
        select(Question, score_expr)
        .where(Question.has_answer == True)
        .order_by(Question.embedding.cosine_distance(vector_query))
        .limit(1)
    )
    
    results = db.exec(statement).all()

    formatted_results = []
    for q_item, score_val in results:
        formatted_results.append(({
            "id": str(q_item.id),
            "question": q_item.question,
            "answer": q_item.answer or "",
            "topic": q_item.topic or "Chưa rõ"
        }, 1.0 - float(score_val) if score_val else 0.0))

    context_dict = build_context(formatted_results)
    context_data = context_dict["context"]

    # 3. Gọi Gemini sinh câu trả lời RAG
    answer = generate_rag_answer(question, context_data, history)

    # 4. Lưu lịch sử hội thoại trực tiếp vào PostgreSQL
    response_type = "answered"
    if "chưa hỗ trợ chủ đề này" in answer:
        response_type = "out_of_topic"
    elif "chưa có thông tin" in answer:
        response_type = "not_found"
    elif context_dict["score"] < 0.7:
        response_type = "auto_generated"

    conversation = Conversation(
        question=question,
        answer=answer,
        response_type=response_type,
        context=context_dict["content"] or None
    )
    db.add(conversation)
    db.commit()

    return {
        "question": question,
        "context": context_dict,
        "answer": answer
    }
