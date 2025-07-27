from collection import vectorstore

def render_data():
    peek_data = vectorstore._collection.get()
    print(peek_data)
    # for i, doc in enumerate(peek_data["documents"]):
    #     metadata = peek_data["metadatas"][i]
    #     print("ID:", peek_data["ids"][i])
    #     print("Document:", doc)
    #     print("Answer:", metadata.get("answer"))
    #     print("—" * 40)

def retrieve_data():
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": {"has_answer": True}},
    )

    results = retriever.invoke("giấy ở đâu?")
    for doc in results:
        print("Document:", doc.page_content)
        print("—" * 40)

def build_context_with_scores(results):
    """
    Tạo context để truyền vào Gemini: có thông tin Hỏi - Trả lời và độ tương đồng (score).
    """
    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
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

def similarity_search():
    results = vectorstore.similarity_search_with_relevance_scores(
        query="Xin giấy xác nhận là sinh viên thì xin ở đâu ạ?",
        # metadata={"has_answer": True},
        k=10,
    )
    for doc, score in results:
        print(score, doc.page_content)
        print("==========================")

def check_duplicate():
    questions = [
        {"question": "Em muốn xin giấy xác nhận là sinh viên thì xin ở đâu ạ?"},
        {"question": "Em muốn làm lại thẻ sinh viên thì làm ở đâu ạ?"},
        {"question": "Em muốn xin chuyển ngành thì cần những giấy tờ gì?"},
    ]

    results = []

    for q in questions:
        question_text = q.get("question", "")

        data = vectorstore.similarity_search_with_relevance_scores(
            query=question_text,
            k=1,
        )

        if data:
            doc, score = data[0]
            print("Metadata:", doc.metadata)
            results.append({
                "id": doc.id,
                "question_dup": doc.page_content,
                "score": round(score * 100, 2),
                "question": question_text
            })
        else:
            results.append({
                "id": None,
                "question_dup": None,
                "score": 0
            })

    print(results)

def delete_embed_file(file_path: str):
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
        print(f"[!] Không tìm thấy đoạn embedding nào từ file: {file_path}")

# render_data()
# retrieve_data()
similarity_search()
# check_duplicate()
# delete_embed_file("documents/EintJ75UYX9HLOilU9F37Mg4EUQDQkgl372AMzdr.pdf")


