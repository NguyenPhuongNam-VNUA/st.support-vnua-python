from collection import get_gemini_embedding_model, get_vectorstore, get_duplicate_questions_vectorstore
import requests
import traceback
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

import re
from langchain.schema import Document

# Load biến môi trường
load_dotenv()

def upsert_from_dict(id: str, embed_data: str, answer: str, has_answer: bool = True, topic: str= ""):
    try:
        # Khởi tạo embedding model
        embedding_model = get_gemini_embedding_model(task_type='RETRIEVAL_DOCUMENT')
        # Lấy vectorstore đã được khởi tạo
        vectorstore = get_vectorstore(embedding_model)

        vectorstore.add_texts(
            texts=[embed_data],
            ids=[id],
            metadatas=[{
                "topic": topic,
                "answer": answer,
                "has_answer": has_answer
            }]
        )
        print(f"[✓] 1, Upsert thành công id: {id}")

        # Lưu vào vectorstore duplicate
        duplicate_vectorstore = get_duplicate_questions_vectorstore()
        duplicate_vectorstore.add_texts(
            texts=[embed_data],
            ids=[id]
        )
        print(f"[✓] 2, Lưu câu hỏi vào vectorstore check duplicate thành công id: {id}")
        return True
    except Exception as e:
        print(f"[✗] Lỗi khi upsert: {e}")
        return False

def delete_from_dict(id: str):
    try:
        # Xóa từ vectorstore RAG
        embedding_model = get_gemini_embedding_model(task_type='RETRIEVAL_DOCUMENT')
        vectorstore = get_vectorstore(embedding_model)
        vectorstore.delete(ids=[id])
        print(f"[✓]1, Xoá thành công id từ vectorstore RAG: {id}")

        # Xoá từ vectorstore duplicate
        duplicate_vectorstore = get_duplicate_questions_vectorstore()
        duplicate_vectorstore.delete(ids=[id])
        print(f"[✓]2, Xoá thành công id từ vectorstore duplicate: {id}")
    except Exception as e:
        print(f"[✗] Lỗi khi xoá: {e}")

def is_duplicate_question(embed_data: str, threshold: float = 0.93):
    try:
        # Embedding câu hỏi để kiểm tra trùng lặp
        embedding_model = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
        vector = embedding_model.embed_query(embed_data)

        # Khởi tạo vectorstore duplicate để kiểm tra câu hỏi trùng lặp
        duplicate_vectorstore = get_duplicate_questions_vectorstore()
        results = duplicate_vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=vector,
            k=1
        )
        print(results)
        if results and ( 1 - results[0][1] ) > threshold:   #results[0][1] là điểm số tương tự (vị trí 1 trong tuple)
            doc, score = results[0]                         # score = results[0][1], doc = results[0][0]
            similarity_score = 1 - score                    # Chuyển đổi sang điểm tương tự (1 - score)
            print(f"[✓] Câu hỏi trùng lặp với điểm số: {similarity_score:.2f}")
            return True, doc.page_content, doc.id, similarity_score
        else:
            print("[✗] Không tìm thấy câu hỏi trùng lặp.")
            return False, None, None, 0
    except Exception as e:
        traceback.print_exc()
        print(f"[✗] Lỗi kiểm tra trùng lặp: {e}")
        return False, None, None, 0

def maybe_save_question_to_db(question: str, answer: str, context_id: str, context_score: float, context_content: str):
    laravel_api = os.getenv("LARAVEL_API_BASE_URL")
    secret = os.getenv("PUBLIC_QUESTION_SECRET")

    # Ngưỡng điểm để chứng minh câu trả lời đáng tin cậy (hay là trả lời chính xác) để + ask_count
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

            # call api Laravel tăng ask_count
            res = requests.post(
                f"{laravel_api}/public/increment-ask-count",
                json={
                    "id": int(current_id),
                },
                headers={"x-api-secret": secret}
            )
            print(f"[✓] Gửi yêu cầu tăng ask_count (phát sinh) cho id: {current_id}")
            print(f"[→] Status Code: {res.status_code}")
        else:
            # call api Laravel store() với table questions --> sẽ có id_new
            res = requests.post(
                f"{laravel_api}/public/questions",
                json={
                    "question": question,
                },
                headers={"x-api-secret": secret}
            )
            print(f"[✓] Gửi yêu cầu lưu câu hỏi: {question}")
            print(f"[→] Status Code: {res.status_code}")

            id_new = str(res.json()["id"])
            current_id = id_new
            # Lưu vào vectorstore duplicate (câu hỏi phát sinh mới)
            duplicate_vectorstore = get_duplicate_questions_vectorstore()
            duplicate_vectorstore.add_texts(
                texts=[question],
                ids=[id_new]
            )
            print(f"[✓] Lưu câu hỏi mới vào vectorstore check duplicate thành công id: {id_new}")

    else:
        if final_score >= relevance_score:
            response_type = "answered"

            # call api Laravel tăng ask_count
            res = requests.post(
                f"{laravel_api}/public/increment-ask-count",
                json={
                    "id": int(current_id),
                },
                headers={"x-api-secret": secret}
            )
            print(f"[✓] Gửi yêu cầu tăng ask_count (đã có) cho id: {current_id}")
            print(f"[→] Status Code: {res.status_code}")
        else:
            response_type = "auto_generated"
            current_id = None
            current_content = None
            print(f"[✓] Câu trả lời tự động sinh bởi LLM.")


    #call api Laravel tạo mới conversations
    res = requests.post(
        f"{laravel_api}/conversations",
        json={
            "question": str(question),
            "answer": str(answer),
            "response_type": str(response_type),
            "context": str(context_content),
        },
        headers={"x-api-secret": secret}
    )
    print(f"[✓] Gửi yêu cầu lưu cuộc hội thoại.")
    if res.status_code == 200 or res.status_code == 201:
        print(f"[→] Tạo mới conversation thành công: {res.status_code}")
    else:
        print("[✗] Lỗi lưu cuộc hội thoại:", res.status_code)
    return


# DOCUMENT EMBEDDING FUNCTIONS
# def update_embedding_status(api_base, file_path, chunk_size, chunk_overlap, auth_token=None):
#     headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
#
#     response = requests.post(
#         f"{api_base}/documents/update-status",
#         json={
#             "file_path": file_path,
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap
#         },
#         headers=headers
#     )
#     print("Laravel update status:", response.status_code, response.text)
#
# def run_embedding_from_file(file_path: str, chunk_size=1000, chunk_overlap=200, token: str = None):
#     try:
#         print(f"[✓] Bắt đầu embedding từ file PDF: {file_path}")
#
#         # Đường dẫn Laravel API
#         api_base = os.getenv("LARAVEL_API_BASE_URL")  # VD: http://localhost:8000
#         url = f"{api_base}/pdf-view?path={file_path}"
#
#         #Tải file PDF từ Laravel API
#         response  = requests.get(url) #Phương thức GET để tải file PDF
#         response.raise_for_status() # Kiểm tra xem có lỗi trong quá trình tải file không
#
#         # Tạo file tạm từ nội dung PDF
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(response.content)
#             tmp_path = tmp_file.name
#
#         # Bây giờ tmp_path là đường dẫn thực tế trên ổ đĩa
#         loader = PyMuPDFLoader(tmp_path)
#         docs = loader.load()
#
#         full_text = "\n".join([doc.page_content for doc in docs])
#
#         # 4. Tìm đề mục dạng La Mã: I. II. III. ...
#         pattern = r"(?:^|\n)([IVXLCDM]+\.\s+[^\n]+)"
#         matches = re.split(pattern, full_text)
#
#         documents = []
#         for i in range(1, len(matches), 2):
#             title = matches[i].strip()
#             content = matches[i + 1].strip()
#
#             # Gán group là đề mục chính (VD: 'I', 'II') để lọc dễ
#             group_code = title.split(".")[0].strip()  # Lấy 'I' từ 'I. THÔNG TIN CHUNG'
#             documents.append(Document(
#                 page_content=content,
#                 metadata={
#                     "title": title,
#                     "groups": group_code
#                 }
#             ))
#
#         # 5. Xoá file tạm
#         os.remove(tmp_path)
#
#         # 6. Chunk từng mục lớn
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n\n", "\n", " ", ""]  # Tách theo đoạn, dòng, khoảng trắng
#         )
#         chunks = splitter.split_documents(documents)
#
#         # 7. Chuẩn bị embedding
#         texts = [chunk.page_content for chunk in chunks]
#         metadatas = [
#             {
#                 "source": file_path,
#                 "title": chunk.metadata.get("title", ""),
#                 "groups": chunk.metadata.get("groups", "")
#             } for chunk in chunks
#         ]
#
#         # 8. Embedding
#         vectorstore.add_texts(texts=texts, metadatas=metadatas)
#         print(f"[✓] Đã embedding {len(texts)} đoạn từ: {file_path}")
#
#         # 9. Laravel update
#         update_embedding_status(api_base, file_path, chunk_size, chunk_overlap, token)
#
#     except Exception as e:
#         print(f"[✗] Lỗi khi embedding file {file_path}: {e}")
#         import traceback
#         traceback.print_exc()
#
# def delete_embed_file(file_path: str):
#     try:
#         # Lấy toàn bộ embedding (metadatas + ids)
#         results = vectorstore._collection.get(include=["metadatas"])
#         ids = results["ids"]
#         metadatas = results["metadatas"]
#
#         # Lọc các đoạn có metadata["source"] == file_path
#         ids_to_delete = [
#             doc_id for doc_id, metadata in zip(ids, metadatas)
#             if metadata.get("source") == file_path
#         ]
#
#         if ids_to_delete:
#             vectorstore.delete(ids=ids_to_delete)
#             print(f"[✓] Đã xoá {len(ids_to_delete)} đoạn embedding từ file: {file_path}")
#         else:
#             print(f"[✗] Không tìm thấy đoạn nào để xoá từ file: {file_path}")
#
#     except Exception as e:
#         print(f"[✗] Lỗi khi xoá embedding từ file {file_path}: {e}")
#         import traceback
#         traceback.print_exc()