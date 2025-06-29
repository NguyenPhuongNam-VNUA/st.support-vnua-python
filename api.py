from flask import Flask, request, jsonify
from flask_cors import CORS

from embed_utils import (
    upsert_from_dict, delete_from_dict,
    is_duplicate_question, maybe_save_question_to_db
)
from collection import vectorstore
from rag_utils import generate_rag_answer
from langchain.chains import RetrievalQA

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

@app.route("/api/embed", methods=["POST"])
def embed_question():
    data = request.json
    upsert_from_dict(
        id=str(data.get("id")),
        question=str(data.get("question")),
        answer=str(data.get("answer")),
        has_answer=(data.get("has_answer"))
    )
    return jsonify({"message": "Embedding thành công"}), 200

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
