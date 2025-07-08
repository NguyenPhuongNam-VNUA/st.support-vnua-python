try:
    import sqlite_patch
except ImportError:
    print("⚠️  Bỏ qua patch SQLite vì thiếu pysqlite3 (chạy local)")

from flask import Flask, request, jsonify
from flask_cors import CORS

from embed_utils import (
    upsert_from_dict, delete_from_dict,
    is_duplicate_question, maybe_save_question_to_db,
    run_embedding_from_file
)
from collection import vectorstore
from rag_utils import generate_rag_answer

app = Flask(__name__)
CORS(app)

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


def build_context_with_scores(results):
    """
    Tạo context để truyền vào Gemini: có thông tin Hỏi - Trả lời và độ tương đồng (score).
    """
    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
        if not doc.metadata.get("has_answer", True):
            continue
        match_percent = score * 100
        context_parts.append(
            f"""---
            KẾT QUẢ #{i}
            Mức độ tương đồng với câu hỏi: {match_percent:.2f}%
            
            • Hỏi: {doc.page_content}
            • Trả lời: {doc.metadata.get('answer', 'Chưa rõ')}
            """
        )
    return "\n".join(context_parts) if context_parts else "Không tìm thấy dữ liệu phù hợp."

@app.route("/api/embed-doc", methods=["POST"])
def embed_doc():
    data = request.json
    file_path = data.get("file_path")

    if not file_path:
        return jsonify({"error": "Thiếu đường dẫn file"}), 400

    # gọi hàm xử lý chunking + embedding tại đây
    run_embedding_from_file(file_path)  # ví dụ
    return jsonify({"message": "Embedding thành công"}), 200

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    results = vectorstore.similarity_search_with_relevance_scores(
        query=question,
        k=3,
        # filter={"has_answer": True}
    )

    context = build_context_with_scores(results)
    answer = generate_rag_answer(question, context)
    maybe_save_question_to_db(question, answer)

    return jsonify({"question": question, "context": context, "answer": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
