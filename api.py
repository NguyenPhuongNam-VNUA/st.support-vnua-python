try:
    import sqlite_patch
except ImportError:
    print("⚠️  Bỏ qua patch SQLite vì thiếu pysqlite3 (chạy local)")

from flask import Flask, request, jsonify
from flask_cors import CORS

from embed_utils import (
    upsert_from_dict, delete_from_dict,
    is_duplicate_question, maybe_save_question_to_db,
    run_embedding_from_file, delete_embed_file
)
from collection import vectorstore
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
    data = request.json
    question = str(data.get("question"))
    is_dup, doc, doc_id, score = is_duplicate_question(question)
    if is_dup:
        return jsonify({
            "is_duplicate": True,
            "message": "Câu hỏi đã tồn tại.",
            "existing_doc": doc,
            "existing_id": doc_id,
            "score_str": f"{score * 100:.2f}%"
        }), 409
    return jsonify({"is_duplicate": False, "message": "Câu hỏi mới."}), 200

@app.route("/api/check-excel", methods=["POST"])
def check_excel():
    data = request.json
    questions = data.get("questions", [])

    results = []
    for question in questions:
        question_text = question.get("question", "")
        answer_text = question.get("answer", "")
        has_answer = question.get("has_answer", False)
        is_dup, doc, doc_id, score = is_duplicate_question(question_text)

        results.append({
            "question": question_text,
            "answer": answer_text,
            "has_answer": has_answer,
            "is_duplicate": is_dup,
            "existing_doc": doc,
            "existing_id": doc_id,
            "score": round(score * 100, 2)
        })
    print("[✓] Thực hiện kiểm tra trùng lặp xong.")
    return jsonify({"results": results}), 200

@app.route("/api/embed", methods=["POST"])
def embed_question():
    data = request.json
    upsert_from_dict(
        id=str(data.get("id")),
        question=str(data.get("question")),
        answer=str(data.get("answer")),
        has_answer=(data.get("has_answer"))
    )

    print("[✓] Thực hiện embedding xong.")
    return jsonify({"message": "Embedding thành công"}), 200

@app.route("/api/embed-batch", methods=["POST"])
def embed_batch():
    # print("→ Nhận yêu cầu embed-batch")
    data = request.json
    questions = data.get("questions", [])
    # print(questions)

    for question in questions:
        # print(question.get("id"), question.get("question"))
        upsert_from_dict(
            id=str(question.get("id")),
            question=str(question.get("question")),
            answer=str(question.get("answer")),
            has_answer=(question.get("has_answer", True))
        )

    print("[✓] Thực hiện embedding batch xong.")
    return jsonify({"message": "Embedding batch thành công", "data": questions}), 200

@app.route("/api/delete-embed", methods=["POST"])
def delete_embed():
    data = request.json
    delete_from_dict(str(data.get("id")))
    return jsonify({"message": "Xoá thành công"}), 200

@app.route("/api/delete-embed-many", methods=["POST"])
def delete_embed_many():
    data = request.json
    ids = data.get("ids", [])

    for _id in ids:
        delete_from_dict(str(_id))

    return jsonify({"message": f"Đã xoá {len(ids)} embedding."}), 200

@app.route("/api/embed-doc", methods=["POST"])
def embed_doc():
    data = request.json
    file_path = data.get("file_path")
    chunk_size = data.get("chunk_size", 1000)
    chunk_overlap = data.get("chunk_overlap", 200)
    token = data.get("token")

    if not file_path:
        return jsonify({"error": "Thiếu đường dẫn file"}), 400

    # gọi hàm xử lý chunking + embedding tại đây
    run_embedding_from_file(file_path, chunk_size, chunk_overlap, token=token)
    return jsonify({"message": "Embedding thành công"}), 200

@app.route("/api/delete-doc", methods=["POST"])
def delete_doc():
    data = request.json
    file_path = data.get("file_path")

    if not file_path:
        return jsonify({"error": "Thiếu đường dẫn file"}), 400

    delete_embed_file(file_path)
    return jsonify({"message": f"Đã xoá embedding từ file: {file_path}"}), 200

def build_context_mixed(results):
    """
    Gộp cả: dữ liệu câu hỏi-có-trả lời (FAQ), và dữ liệu file PDF có title, groups.
    """
    faq_parts = []
    pdf_parts = []
    seen_groups = set()

    for i, (doc, score) in enumerate(results, 1):
        # Nếu là câu hỏi - câu trả lời có sẵn (FAQ)
        if doc.metadata.get("has_answer", False):
            match_percent = score * 100
            faq_parts.append(
                f"""---\nKẾT QUẢ #{i} - Mức độ tương đồng với câu hỏi: {match_percent:.2f}%\n\n• Hỏi: {doc.page_content}\n• Trả lời: {doc.metadata.get('answer', 'Chưa rõ')}\n"""
            )

        # Nếu là tài liệu từ file PDF
        elif doc.metadata.get("source"):
            group = doc.metadata.get("groups", "")
            title = doc.metadata.get("title", "")

            # Nếu gặp chương mới, thêm heading
            if group not in seen_groups:
                pdf_parts.append(f"\n### {title}\n")
                seen_groups.add(group)

            pdf_parts.append(doc.page_content.strip())

    # Ưu tiên: FAQ lên đầu → rồi context từ file PDF
    context = ""
    if faq_parts:
        context += "### Câu hỏi - trả lời đã có trong hệ thống:\n" + "\n".join(faq_parts)

    if pdf_parts:
        context += "\n\n### Ngữ cảnh trích từ tài liệu:\n" + "\n".join(pdf_parts)

    return context.strip() if context else "Không tìm thấy dữ liệu phù hợp."

@app.route("/api/ask", methods=["POST"])
def ask():
    start = time.time()
    data = request.json
    question = data.get("question")
    history = data.get("messages", [])

    t1 = time.time()
    results = vectorstore.similarity_search_with_relevance_scores(
        query=question,
        k=10,
    )
    print(f"[⏱️] similarity_search: {time.time() - t1:.2f}s")

    t2 = time.time()
    context =build_context_mixed(results)
    # return jsonify({ "context": context}), 200
    answer = generate_rag_answer(question, context, history)
    print(f"[⏱️] Gemini answer: {time.time() - t2:.2f}s")

    t3 = time.time()
    maybe_save_question_to_db(question, answer)
    print(f"[⏱️] Save DB: {time.time() - t3:.2f}s")

    print(f"[✅] Tổng thời gian xử lý: {time.time() - start:.2f}s")

    return jsonify({"question": question, "context": context, "answer": answer})

if __name__ == "__main__":
    app.run(port=5000)
