from collection import vectorstore
import requests
import traceback
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from dotenv import load_dotenv

import re
from langchain.schema import Document

# Load biến môi trường
load_dotenv()

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
            laravel_api = os.getenv("LARAVEL_API_BASE_URL")
            secret = os.getenv("PUBLIC_QUESTION_SECRET")

            res = requests.post(
                f"{laravel_api}/public/questions",
                json={
                    "question": question,
                    "answer": None,
                    "has_answer": False
                },
                headers={"x-api-secret": secret}
            )

            print(f"[✓] Gửi yêu cầu lưu câu hỏi: {question}")
            print(f"[→] Status Code: {res.status_code}")

            if res.status_code == 200 or res.status_code == 201:
                print(f"[✓] Đã lưu câu hỏi chưa có thông tin: {question}")
            else:
                print(f"[✗] Laravel trả về lỗi: {res.status_code}")

        except Exception as e:
            print(f"[✗] Lỗi khi gửi request tới Laravel: {e}")

def update_embedding_status(api_base, file_path, chunk_size, chunk_overlap, auth_token=None):
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}

    response = requests.post(
        f"{api_base}/documents/update-status",
        json={
            "file_path": file_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        },
        headers=headers
    )
    print("Laravel update status:", response.status_code, response.text)

def run_embedding_from_file(file_path: str, chunk_size=1000, chunk_overlap=200, token: str = None):
    try:
        print(f"[✓] Bắt đầu embedding từ file PDF: {file_path}")

        # Đường dẫn Laravel API
        api_base = os.getenv("LARAVEL_API_BASE_URL")  # VD: http://localhost:8000
        url = f"{api_base}/pdf-view?path={file_path}"

        #Tải file PDF từ Laravel API
        response  = requests.get(url) #Phương thức GET để tải file PDF
        response.raise_for_status() # Kiểm tra xem có lỗi trong quá trình tải file không

        # Tạo file tạm từ nội dung PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        # Bây giờ tmp_path là đường dẫn thực tế trên ổ đĩa
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        full_text = "\n".join([doc.page_content for doc in docs])

        # 4. Tìm đề mục dạng La Mã: I. II. III. ...
        pattern = r"(?:^|\n)([IVXLCDM]+\.\s+[^\n]+)"
        matches = re.split(pattern, full_text)

        documents = []
        for i in range(1, len(matches), 2):
            title = matches[i].strip()
            content = matches[i + 1].strip()

            # Gán group là đề mục chính (VD: 'I', 'II') để lọc dễ
            group_code = title.split(".")[0].strip()  # Lấy 'I' từ 'I. THÔNG TIN CHUNG'
            documents.append(Document(
                page_content=content,
                metadata={
                    "title": title,
                    "groups": group_code
                }
            ))

        # 5. Xoá file tạm
        os.remove(tmp_path)

        # 6. Chunk từng mục lớn
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Tách theo đoạn, dòng, khoảng trắng
        )
        chunks = splitter.split_documents(documents)

        # 7. Chuẩn bị embedding
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": file_path,
                "title": chunk.metadata.get("title", ""),
                "groups": chunk.metadata.get("groups", "")
            } for chunk in chunks
        ]

        # 8. Embedding
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"[✓] Đã embedding {len(texts)} đoạn từ: {file_path}")

        # 9. Laravel update
        update_embedding_status(api_base, file_path, chunk_size, chunk_overlap, token)

    except Exception as e:
        print(f"[✗] Lỗi khi embedding file {file_path}: {e}")
        import traceback
        traceback.print_exc()

def delete_embed_file(file_path: str):
    try:
        # Lấy toàn bộ embedding (metadatas + ids)
        results = vectorstore._collection.get(include=["metadatas"])
        ids = results["ids"]
        metadatas = results["metadatas"]

        # Lọc các đoạn có metadata["source"] == file_path
        ids_to_delete = [
            doc_id for doc_id, metadata in zip(ids, metadatas)
            if metadata.get("source") == file_path
        ]

        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            print(f"[✓] Đã xoá {len(ids_to_delete)} đoạn embedding từ file: {file_path}")
        else:
            print(f"[✗] Không tìm thấy đoạn nào để xoá từ file: {file_path}")

    except Exception as e:
        print(f"[✗] Lỗi khi xoá embedding từ file {file_path}: {e}")
        import traceback
        traceback.print_exc()