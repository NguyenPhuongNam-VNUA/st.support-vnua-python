try:
    import sqlite_patch
except ImportError:
    print("⚠️  Bỏ qua patch SQLite vì thiếu pysqlite3 (chạy local)")

from flask import Flask, request, jsonify
from flask_cors import CORS
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
import time
app = Flask(__name__)
CORS(
    app,
    origins=[
        "https://st-dse.vnua.edu.vn:6896",
        "http://127.0.0.1:5173",
        "http://localhost:5173"],
    supports_credentials=True
)

@app.route("/api/check-duplicate", methods=["POST"])
def check_duplicate():
    # Lấy dữ liệu từ request gửi từ react -> api laravel store
    data = request.json
    question = str(data.get("question"))
    related_questions = data.get("related_questions")

    # Làm sạch dữ liệu embedding
    if not related_questions:
        embed_data = question.strip()
    else:
        embed_data = f"{question}\n{related_questions}".strip()

    is_dup, doc, doc_id, score = is_duplicate_question(embed_data)
    if is_dup:
        return jsonify({
            "is_duplicate": True,
            "message": "Câu hỏi đã tồn tại.",
            "existing_doc": doc,
            "existing_id": doc_id,
            "score_str": f"{score * 100:.2f}%"
        }), 409
    return jsonify({"is_duplicate": False, "message": "Câu hỏi mới."}), 200

# @app.route("/api/check-excel", methods=["POST"])
# def check_excel():
#     data = request.json
#     questions = data.get("questions", [])
#
#     results = []
#     for question in questions:
#         question_text = question.get("question", "")
#         answer_text = question.get("answer", "")
#         has_answer = question.get("has_answer", False)
#         is_dup, doc, doc_id, score = is_duplicate_question(question_text)
#
#         results.append({
#             "question": question_text,
#             "answer": answer_text,
#             "has_answer": has_answer,
#             "is_duplicate": is_dup,
#             "existing_doc": doc,
#             "existing_id": doc_id,
#             "score": round(score * 100, 2)
#         })
#     print("[✓] Thực hiện kiểm tra trùng lặp xong.")
#     return jsonify({"results": results}), 200

@app.route("/api/embed", methods=["POST"])
def embed_question():
    # Lấy dữ liệu từ request gửi từ react -> api laravel store
    data = request.json
    id = data.get("id")
    question = data.get("question", "")
    answer = data.get("answer", "")
    has_answer = data.get("has_answer")
    related_questions = data.get("related_questions","")
    topic = data.get("topic", "")

    # Làm sạch dữ liệu embedding
    if not related_questions:
        embed_data = question.strip()
    else:
        embed_data = f"{question}\n{related_questions}".strip()

    try:
        is_embed = upsert_from_dict(
            id=str(id),
            embed_data=embed_data,
            answer=str(answer).strip(),
            has_answer=has_answer,
            topic=str(topic).strip()
        )
        print("[✓] Thực hiện embedding xong.")
        return jsonify({"message": "Embedding thành công", "is_embed": is_embed}), 200
    except Exception as e:
        print(f"[✗] API lỗi: {e}")
        return jsonify({"message": "Embedding thất bại", "error": str(e)}), 500

@app.route("/api/countToken", methods=["POST"])
def count_input_tokens():
   data = request.json
   embeddings_texts = str(data.get("text", ""))
   tokenize = CustomGeminiEmbeddings.get_input_token_count(embeddings_texts)
   return jsonify({"token_count": tokenize}), 200


@app.route("/api/embed-batch", methods=["POST"])
def embed_batch():
    # print("→ Nhận yêu cầu embed-batch")
    data = request.json
    questions = data.get("questions", [])
    # print(questions)

    for question in questions:
        if not question.get("related_questions"):
            embed_data = question.get("question", "").strip()
        else:
            embed_data = f"{question.get('question','').strip()}\n{question.get('related_questions','').strip()}".strip()
        upsert_from_dict(
            id=str(question.get("id")),
            embed_data=embed_data,
            answer=str(question.get("answer")),
            has_answer=(question.get("has_answer", True)),
            topic=str(question.get("topic", "")),
        )

    print("[✓] Thực hiện embedding batch xong.")
    return jsonify({"message": "Embedding batch thành công"}), 200

@app.route("/api/delete-embed", methods=["POST"])
def delete_embed():
    data = request.json
    delete_from_dict(str(data.get("id")))
    return jsonify({"message": "Xoá embed thành công"}), 200

@app.route("/api/delete-embed-many", methods=["POST"])
def delete_embed_many():
    data = request.json
    ids = data.get("ids", [])

    for _id in ids:
        delete_from_dict(str(_id))

    return jsonify({"message": f"Đã xoá {len(ids)} embedding."}), 200

# @app.route("/api/embed-doc", methods=["POST"])
# def embed_doc():
#     data = request.json
#     file_path = data.get("file_path")
#     chunk_size = data.get("chunk_size", 1000)
#     chunk_overlap = data.get("chunk_overlap", 200)
#     token = data.get("token")
#
#     if not file_path:
#         return jsonify({"error": "Thiếu đường dẫn file"}), 400
#
#     # gọi hàm xử lý chunking + embedding tại đây
#     run_embedding_from_file(file_path, chunk_size, chunk_overlap, token=token)
#     return jsonify({"message": "Embedding thành công"}), 200

# @app.route("/api/delete-doc", methods=["POST"])
# def delete_doc():
#     data = request.json
#     file_path = data.get("file_path")
#
#     if not file_path:
#         return jsonify({"error": "Thiếu đường dẫn file"}), 400
#
#     delete_embed_file(file_path)
#     return jsonify({"message": f"Đã xoá embedding từ file: {file_path}"}), 200

# def build_context_mixed(results):
#     """
#     Gộp cả: dữ liệu câu hỏi-có-trả lời (FAQ), và dữ liệu file PDF có title, groups.
#     """
#     faq_parts = []
#     pdf_parts = []
#     seen_groups = set()
#
#     for i, (doc, score) in enumerate(results, 1):
#         # Nếu là câu hỏi - câu trả lời có sẵn (FAQ)
#         if doc.metadata.get("has_answer", False):
#             match_percent = score * 100
#             faq_parts.append(
#                 f"""---\n[Thông tin tham khảo #{i}] \n• Hỏi: {doc.page_content}\n• Trả lời: {doc.metadata.get('answer', 'Chưa rõ')}\n"""
#             )
#
#         # Nếu là tài liệu từ file PDF
#         elif doc.metadata.get("source"):
#             group = doc.metadata.get("groups", "")
#             title = doc.metadata.get("title", "")
#
#             # Nếu gặp chương mới, thêm heading
#             if group not in seen_groups:
#                 pdf_parts.append(f"\n### {title}\n")
#                 seen_groups.add(group)
#
#             pdf_parts.append(doc.page_content.strip())
#
#     # Ưu tiên: FAQ lên đầu → rồi context từ file PDF
#     context = ""
#     if faq_parts:
#         context += "### Câu hỏi - trả lời đã có trong hệ thống:\n" + "\n".join(faq_parts)
#
#     if pdf_parts:
#         context += "\n\n### Ngữ cảnh trích từ tài liệu:\n" + "\n".join(pdf_parts)
#
#     return context.strip() if context else "Không tìm thấy dữ liệu phù hợp."

# sửa nhanh với k=1 nên không quan trọng mảng
def build_context(results):
    context_parts = []
    score = 0
    id = ""
    content = ""

    for i, (doc, score) in enumerate(results, 1):
        topic = doc.metadata.get("topic", "Chưa rõ")
        answer = doc.metadata.get("answer", "Chưa có câu trả lời")
        question = doc.page_content.strip()

        # Lấy dữ liệu để lưu vào maybe_save_question_to_db
        id = doc.id
        score = score
        content = question

        context_parts.append(
            f"[Thông tin tham khảo #{i}]:\n"
            f"- Chủ đề: {topic}\n"
            f"- Hỏi: {question}\n"
            f"- Trả lời: {answer}\n"
        )
    return {
        "context": "\n".join(context_parts).strip(),
        "score": 1-score,
        "id": id,
        "content": content
    }

@app.route("/api/ask", methods=["POST"])
def ask():
    # start = time.time()
    data = request.json
    question = data.get("question")
    history = data.get("messages", [])
    # print(question,history)
    # t1 = time.time()
    # Lấy vector
    embedding_model = get_gemini_embedding_model("RETRIEVAL_QUERY")
    vector_query = embedding_model.embed_query(question)

    # Tìm kiếm tương tự
    model = get_gemini_embedding_model("RETRIEVAL_DOCUMENT")
    vectorstore = get_vectorstore(model)
    results = vectorstore.similarity_search_by_vector_with_relevance_scores(
        embedding=vector_query,
        k=1,
    )
    # print(f"[⏱️] similarity_search: {time.time() - t1:.2f}s")
    # doc, score = results[0]
    # print(results)
    # return jsonify({
    #     "question": question,
    #     "context": doc.page_content,
    #     "answer": doc.metadata.get("answer", "Chưa có câu trả lời"),
    #     "topic": doc.metadata.get("topic", "Chưa rõ"),
    #     "score": f"{(1 - score) * 100:.2f}%"
    # })
    # t2 = time.time()
    context = build_context(results)
    context_data = context["context"]
    # print(context)
    # return jsonify({ "context": context}), 200
    answer = generate_rag_answer(question, context_data, history)
    # print(f"[⏱️] Gemini answer: {time.time() - t2:.2f}s")

    # t3 = time.time()
    maybe_save_question_to_db(question, answer, context["id"], context["score"], context["content"])
    # print(f"[⏱️] Save DB: {time.time() - t3:.2f}s")

    # print(f"[✅] Tổng thời gian xử lý: {time.time() - start:.2f}s")

    return jsonify({"question": question, "context": context, "answer": answer})

if __name__ == "__main__":
    app.run(port=5000)
