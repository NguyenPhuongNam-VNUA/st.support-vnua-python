from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from sqlmodel import Session, select
from app.schemas.question import (
    CheckDuplicateRequest,
    EmbedRequest,
    EmbedBatchRequest,
    DeleteEmbedRequest,
    DeleteEmbedManyRequest
)
from app.core.database import get_session
from app.models.question import Question
from app.services.embedding_service import CustomGeminiEmbeddings

router = APIRouter()

@router.post("/check-duplicate")
async def check_duplicate(payload: CheckDuplicateRequest, db: Session = Depends(get_session)):
    question = payload.question
    related = payload.related_questions

    embed_data = question.strip() if not related else f"{question}\n{related}".strip()

    # Embedding câu hỏi kiểm tra trùng lặp
    embedding_service = CustomGeminiEmbeddings(task_type="SEMANTIC_SIMILARITY")
    vector_query = embedding_service.embed_query(embed_data)

    # Search trên PostgreSQL với pgvector
    score_expr = (1 - Question.embedding.cosine_distance(vector_query)).label("score")
    statement = (
        select(Question, score_expr)
        .order_by(Question.embedding.cosine_distance(vector_query))
        .limit(1)
    )
    
    results = db.exec(statement).first()
    threshold = 0.93

    if results:
        existing_doc, score_val = results
        similarity_score = float(score_val)
        if similarity_score > threshold:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "is_duplicate": True,
                    "message": "Câu hỏi đã tồn tại.",
                    "existing_doc": existing_doc.question,
                    "existing_id": str(existing_doc.id),
                    "score_str": f"{similarity_score * 100:.2f}%"
                }
            )

    return {"is_duplicate": False, "message": "Câu hỏi mới."}

@router.post("/embed")
async def embed_question(payload: EmbedRequest, db: Session = Depends(get_session)):
    embed_data = payload.question.strip() if not payload.related_questions else f"{payload.question}\n{payload.related_questions}".strip()

    try:
        embedding_service = CustomGeminiEmbeddings(task_type="RETRIEVAL_DOCUMENT")
        doc_vector = embedding_service.embed_query(embed_data)

        # Kiểm tra xem ID đã tồn tại trong DB chưa
        db_id = int(payload.id) if str(payload.id).isdigit() else None
        q_item = db.get(Question, db_id) if db_id else None

        if not q_item:
            q_item = Question(
                id=db_id,
                question=payload.question,
                answer=payload.answer,
                has_answer=payload.has_answer if payload.has_answer is not None else True,
                topic=payload.topic or "Chưa phân loại",
                related_questions=payload.related_questions,
                embedding=doc_vector,
                is_embed=True
            )
            db.add(q_item)
        else:
            q_item.question = payload.question
            q_item.answer = payload.answer
            q_item.has_answer = payload.has_answer if payload.has_answer is not None else True
            q_item.topic = payload.topic or q_item.topic
            q_item.related_questions = payload.related_questions
            q_item.embedding = doc_vector
            q_item.is_embed = True

        db.commit()
        return {"message": "Embedding thành công", "is_embed": True}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding thất bại: {str(e)}"
        )

@router.post("/embed-batch")
async def embed_batch(payload: EmbedBatchRequest, db: Session = Depends(get_session)):
    embedding_service = CustomGeminiEmbeddings(task_type="RETRIEVAL_DOCUMENT")

    try:
        for item in payload.questions:
            embed_data = item.question.strip() if not item.related_questions else f"{item.question.strip()}\n{item.related_questions.strip()}".strip()
            doc_vector = embedding_service.embed_query(embed_data)

            db_id = int(item.id) if str(item.id).isdigit() else None
            q_item = db.get(Question, db_id) if db_id else None

            if not q_item:
                q_item = Question(
                    id=db_id,
                    question=item.question,
                    answer=item.answer,
                    has_answer=item.has_answer if item.has_answer is not None else True,
                    topic=item.topic or "Chưa phân loại",
                    related_questions=item.related_questions,
                    embedding=doc_vector,
                    is_embed=True
                )
                db.add(q_item)
            else:
                q_item.question = item.question
                q_item.answer = item.answer
                q_item.has_answer = item.has_answer if item.has_answer is not None else True
                q_item.topic = item.topic or q_item.topic
                q_item.related_questions = item.related_questions
                q_item.embedding = doc_vector
                q_item.is_embed = True

        db.commit()
        return {"message": "Embedding batch thành công"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding batch thất bại: {str(e)}"
        )

@router.post("/delete-embed")
async def delete_embed(payload: DeleteEmbedRequest, db: Session = Depends(get_session)):
    db_id = int(payload.id) if str(payload.id).isdigit() else None
    if db_id:
        q_item = db.get(Question, db_id)
        if q_item:
            q_item.embedding = None
            q_item.is_embed = False
            db.commit()
    return {"message": "Xoá embed thành công"}

@router.post("/delete-embed-many")
async def delete_embed_many(payload: DeleteEmbedManyRequest, db: Session = Depends(get_session)):
    for item_id in payload.ids:
        db_id = int(item_id) if str(item_id).isdigit() else None
        if db_id:
            q_item = db.get(Question, db_id)
            if q_item:
                q_item.embedding = None
                q_item.is_embed = False
    db.commit()
    return {"message": f"Đã xoá {len(payload.ids)} embedding."}
