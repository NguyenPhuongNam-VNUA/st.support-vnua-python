from collection import vectorstore
import requests
import traceback

def upsert_from_dict(id: str, question: str, answer: str, has_answer: bool = True):
    try:
        vectorstore.add_texts(
            texts=[question],
            ids=[id],
            metadatas=[{
                "question": question,
                "answer": answer,
                "has_answer": has_answer
            }]
        )
        print(f"[✓] Upsert thành công: {id}")
    except Exception as e:
        print(f"[✗] Lỗi khi upsert: {e}")

def delete_from_dict(id: str):
    try:
        vectorstore.delete(ids=[id])
        print(f"[✓] Xoá thành công id: {id}")
    except Exception as e:
        print(f"[✗] Lỗi khi xoá: {e}")

def is_duplicate_question(question: str, threshold: float = 0.85):
    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query=question,
            k=1,
        )
        # print(results[0])
        if results and results[0][1] > threshold:
            doc, score = results[0]
            print(f"[✓] Câu hỏi trùng lặp với điểm số: {score:.2f}")
            return True, doc.page_content, doc.metadata.get("id"), score
        else:
            print("[✗] Không tìm thấy câu hỏi trùng lặp.")
            return False, None, None, 0
    except Exception as e:
        traceback.print_exc()
        print(f"[✗] Lỗi kiểm tra trùng lặp: {e}")
        return False, None, None, None

def maybe_save_question_to_db(question: str, answer: str):
    if "chưa có thông tin" in answer:
        try:
            res = requests.post("http://127.0.0.1:8000/api/questions", json={
                "question": question,
                "answer": None,
                "has_answer": False
            })
            if res.status_code == 200:
                print(f"[✓] Đã lưu câu hỏi chưa có thông tin: {question}")
            else:
                print(f"[✗] Câu hỏi đã tồn tại.")
        except Exception as e:
            print(f"[✗] Lỗi khi lưu câu hỏi: {e}")
