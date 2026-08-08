try:
    import sqlite_patch
except ImportError:
    print("⚠️ Bỏ qua patch SQLite vì thiếu pysqlite3 (chạy local)")

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn
from embed_utils import (
    upsert_from_dict, delete_from_dict,
    is_duplicate_question, maybe_save_question_to_db,
)
from gemini_embedding_001.CustomGeminiEmbeddings import CustomGeminiEmbeddings
from collection import (
    get_gemini_embedding_model,
    get_vectorstore,
)
from rag_utils import generate_rag_answer

app = FastAPI(
    title="VNUA Chatbot RAG API",
    description="FastAPI Microservice cho RAG Chatbot Học viện Nông nghiệp Việt Nam",
    version="2.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
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

def build_context(results):
    context_parts = []
    score = 0.0
    id_val = ""
    content = ""

    if results and results.get("ids") and len(results["ids"][0]) > 0:
        id_val = results["ids"][0][0]
        content = results["documents"][0][0]
        distance = results["distances"][0][0]
        score = 1.0 - distance
        metadata = results["metadatas"][0][0] if results.get("metadatas") else {}
        topic = metadata.get("topic", "Chưa rõ")
        answer = metadata.get("answer", "Chưa có câu trả lời")

        context_parts.append(
            f"[Thông tin tham khảo #1]:\n"
            f"- Chủ đề: {topic}\n"
            f"- Hỏi: {content}\n"
            f"- Trả lời: {answer}\n"
        )

    return {
        "context": "\n".join(context_parts).strip(),
        "score": score,
        "id": id_val,
        "content": content
    }

@app.post("/api/check-duplicate")
async def check_duplicate(payload: CheckDuplicateRequest):
    question = payload.question
    related_questions = payload.related_questions

    if not related_questions:
        embed_data = question.strip()
    else:
        embed_data = f"{question}\n{related_questions}".strip()

    is_dup, doc, doc_id, score = is_duplicate_question(embed_data)
    if is_dup:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "is_duplicate": True,
                "message": "Câu hỏi đã tồn tại.",
                "existing_doc": doc,
                "existing_id": doc_id,
                "score_str": f"{score * 100:.2f}%"
            }
        )
    return {"is_duplicate": False, "message": "Câu hỏi mới."}

@app.post("/api/embed")
async def embed_question(payload: EmbedRequest):
    if not payload.related_questions:
        embed_data = payload.question.strip()
    else:
        embed_data = f"{payload.question}\n{payload.related_questions}".strip()

    try:
        is_embed = upsert_from_dict(
            id=str(payload.id),
            embed_data=embed_data,
            answer=str(payload.answer or "").strip(),
            has_answer=payload.has_answer,
            topic=str(payload.topic or "").strip()
        )
        print("[✓] Thực hiện embedding xong.")
        return {"message": "Embedding thành công", "is_embed": is_embed}
    except Exception as e:
        print(f"[✗] API lỗi: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding thất bại: {str(e)}"
        )

@app.post("/api/countToken")
async def count_input_tokens(payload: CountTokenRequest):
    tokenize = CustomGeminiEmbeddings.get_input_token_count(payload.text)
    return {"token_count": tokenize}

@app.post("/api/embed-batch")
async def embed_batch(payload: EmbedBatchRequest):
    for item in payload.questions:
        if not item.related_questions:
            embed_data = item.question.strip()
        else:
            embed_data = f"{item.question.strip()}\n{item.related_questions.strip()}".strip()
        upsert_from_dict(
            id=str(item.id),
            embed_data=embed_data,
            answer=str(item.answer or ""),
            has_answer=item.has_answer if item.has_answer is not None else True,
            topic=str(item.topic or ""),
        )

    print("[✓] Thực hiện embedding batch xong.")
    return {"message": "Embedding batch thành công"}

@app.post("/api/delete-embed")
async def delete_embed(payload: DeleteEmbedRequest):
    delete_from_dict(str(payload.id))
    return {"message": "Xoá embed thành công"}

@app.post("/api/delete-embed-many")
async def delete_embed_many(payload: DeleteEmbedManyRequest):
    for _id in payload.ids:
        delete_from_dict(str(_id))

    return {"message": f"Đã xoá {len(payload.ids)} embedding."}

@app.post("/api/ask")
async def ask(payload: AskRequest):
    question = payload.question
    history = payload.messages or []

    # Lấy vector query
    embedding_model = get_gemini_embedding_model("RETRIEVAL_QUERY")
    vector_query = embedding_model.embed_query(question)

    # Tìm kiếm tương tự trên ChromaDB
    qa_collection = get_vectorstore()
    results = qa_collection.query(
        query_embeddings=[vector_query],
        n_results=1
    )

    context = build_context(results)
    context_data = context["context"]

    # Gọi Gemini sinh câu trả lời RAG
    answer = generate_rag_answer(question, context_data, history)

    # Kiểm tra & lưu vào DB Laravel nếu cần
    maybe_save_question_to_db(question, answer, context["id"], context["score"], context["content"])

    return {"question": question, "context": context, "answer": answer}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5001, reload=True)
