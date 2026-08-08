from collection import get_gemini_embedding_model, get_vectorstore, get_duplicate_questions_vectorstore
import requests
import traceback
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

def upsert_from_dict(id: str, embed_data: str, answer: str, has_answer: bool = True, topic: str = ""):
    try:
        # Khởi tạo embedding model cho tài liệu
        embedding_model = get_gemini_embedding_model(task_type='RETRIEVAL_DOCUMENT')
        doc_vector = embedding_model.embed_query(embed_data)

        # Lấy vectorstore qa_rag_collection
        vectorstore = get_vectorstore()
        vectorstore.upsert(
            ids=[str(id)],
            documents=[embed_data],
            embeddings=[doc_vector],
            metadatas=[{
                "topic": topic,
                "answer": answer,
                "has_answer": has_answer
            }]
        )
        print(f"[✓] 1, Upsert thành công id: {id}")

        # Lưu vào vectorstore duplicate câu hỏi
        duplicate_vectorstore = get_duplicate_questions_vectorstore()
        sim_embedding_model = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
        dup_vector = sim_embedding_model.embed_query(embed_data)
        duplicate_vectorstore.upsert(
            ids=[str(id)],
            documents=[embed_data],
            embeddings=[dup_vector]
        )
        print(f"[✓] 2, Lưu câu hỏi vào vectorstore check duplicate thành công id: {id}")
        return True
    except Exception as e:
        print(f"[✗] Lỗi khi upsert: {e}")
        return False

def delete_from_dict(id: str):
    try:
        # Xóa từ vectorstore RAG
        vectorstore = get_vectorstore()
        vectorstore.delete(ids=[str(id)])
        print(f"[✓] 1, Xoá thành công id từ vectorstore RAG: {id}")

        # Xoá từ vectorstore duplicate
        duplicate_vectorstore = get_duplicate_questions_vectorstore()
        duplicate_vectorstore.delete(ids=[str(id)])
        print(f"[✓] 2, Xoá thành công id từ vectorstore duplicate: {id}")
    except Exception as e:
        print(f"[✗] Lỗi khi xoá: {e}")

def is_duplicate_question(embed_data: str, threshold: float = 0.93):
    try:
        embedding_model = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
        vector = embedding_model.embed_query(embed_data)

        duplicate_vectorstore = get_duplicate_questions_vectorstore()
        results = duplicate_vectorstore.query(
            query_embeddings=[vector],
            n_results=1
        )
        
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            doc_id = results["ids"][0][0]
            doc_content = results["documents"][0][0]
            distance = results["distances"][0][0]
            similarity_score = 1.0 - distance
            print(f"[✓] Điểm tương đồng: {similarity_score:.2f}")
            if similarity_score > threshold:
                return True, doc_content, doc_id, similarity_score
        
        print("[✗] Không tìm thấy câu hỏi trùng lặp.")
        return False, None, None, 0
    except Exception as e:
        traceback.print_exc()
        print(f"[✗] Lỗi kiểm tra trùng lặp: {e}")
        return False, None, None, 0

def maybe_save_question_to_db(question: str, answer: str, context_id: str, context_score: float, context_content: str):
    laravel_api = os.getenv("LARAVEL_API_BASE_URL")
    secret = os.getenv("PUBLIC_QUESTION_SECRET")

    relevance_score = 0.7
    response_type = ""
    current_id = context_id
    current_content = context_content
    final_score = context_score

    if "chưa hỗ trợ chủ đề này" in answer:
        response_type = "out_of_topic"
        current_id = None
        current_content = None
    elif "chưa có thông tin" in answer:
        response_type = "not_found"
        is_dup, doc, doc_id, score = is_duplicate_question(question)
        if is_dup:
            current_id = doc_id
            if laravel_api and secret:
                try:
                    res = requests.post(
                        f"{laravel_api}/public/increment-ask-count",
                        json={"id": int(current_id)},
                        headers={"x-api-secret": secret},
                        timeout=5
                    )
                    print(f"[✓] Gửi yêu cầu tăng ask_count (phát sinh) cho id: {current_id}")
                except Exception as ex:
                    print(f"[✗] Lỗi gọi API increment-ask-count: {ex}")
        else:
            if laravel_api and secret:
                try:
                    res = requests.post(
                        f"{laravel_api}/public/questions",
                        json={"question": question},
                        headers={"x-api-secret": secret},
                        timeout=5
                    )
                    if res.status_code in [200, 201]:
                        id_new = str(res.json()["id"])
                        current_id = id_new
                        duplicate_vectorstore = get_duplicate_questions_vectorstore()
                        sim_embedding_model = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
                        dup_vector = sim_embedding_model.embed_query(question)
                        duplicate_vectorstore.upsert(
                            ids=[id_new],
                            documents=[question],
                            embeddings=[dup_vector]
                        )
                        print(f"[✓] Lưu câu hỏi mới vào vectorstore check duplicate thành công id: {id_new}")
                except Exception as ex:
                    print(f"[✗] Lỗi gọi API save question: {ex}")
    else:
        if final_score >= relevance_score:
            response_type = "answered"
            if current_id and laravel_api and secret:
                try:
                    res = requests.post(
                        f"{laravel_api}/public/increment-ask-count",
                        json={"id": int(current_id)},
                        headers={"x-api-secret": secret},
                        timeout=5
                    )
                    print(f"[✓] Gửi yêu cầu tăng ask_count cho id: {current_id}")
                except Exception as ex:
                    print(f"[✗] Lỗi gọi API increment-ask-count: {ex}")
        else:
            response_type = "auto_generated"
            current_id = None
            current_content = None
            print(f"[✓] Câu trả lời tự động sinh bởi LLM.")

    if laravel_api and secret:
        try:
            res = requests.post(
                f"{laravel_api}/conversations",
                json={
                    "question": str(question),
                    "answer": str(answer),
                    "response_type": str(response_type),
                    "context": str(context_content or ""),
                },
                headers={"x-api-secret": secret},
                timeout=5
            )
            print(f"[✓] Gửi yêu cầu lưu cuộc hội thoại: status {res.status_code}")
        except Exception as ex:
            print(f"[✗] Lỗi lưu cuộc hội thoại: {ex}")
    return