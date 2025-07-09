from collection import vectorstore
import requests
import traceback
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

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
            return True, doc.page_content, doc.id, score
        else:
            print("[✗] Không tìm thấy câu hỏi trùng lặp.")
            return False, None, None, 0
    except Exception as e:
        traceback.print_exc()
        print(f"[✗] Lỗi kiểm tra trùng lặp: {e}")
        return False, None, None, 0

def maybe_save_question_to_db(question: str, answer: str):
    if "chưa có thông tin" in answer:
        try:
            laravel_api =os.getenv("LARAVEL_API_BASE_URL")
            res = requests.post(f"{laravel_api}/questions", json={
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

def run_embedding_from_file(file_path: str):
    try:
        print(f"[✓] Bắt đầu embedding từ file PDF: {file_path}")

        # Thư mục gốc public của Laravel (nơi storage link tới public)
        PUBLIC_STORAGE_ROOT = "/opt/homebrew/var/www/st.support-laravel/public"

        # Ghép đường dẫn tuyệt đối
        abs_path = os.path.join(PUBLIC_STORAGE_ROOT, file_path)
        if not os.path.isfile(abs_path):
            print(f"[✗] File không tồn tại: {abs_path}")
            return

        # Load nội dung file PDF thành Document list
        loader = PyMuPDFLoader(abs_path)
        documents = loader.load()

        # Chia thành các đoạn nhỏ (chunk)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        # print(chunks)
        # Lấy nội dung văn bản từ mỗi chunk
        texts = [chunk.page_content for chunk in chunks]

        # Gắn metadata cho mỗi đoạn (ở đây là nguồn gốc tài liệu)
        metadatas = [{"source": file_path} for _ in texts]

        # Thêm vào vector database
        vectorstore.add_texts(texts=texts, metadatas=metadatas)

        print(f"[✓] Đã embedding {len(texts)} đoạn từ: {file_path}")

    except Exception as e:
        print(f"[✗] Lỗi khi embedding file {file_path}: {e}")
        import traceback
        traceback.print_exc()