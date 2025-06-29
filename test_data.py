from collection import vectorstore

def render_data():
    peek_data = vectorstore._collection.peek()
    for i, doc in enumerate(peek_data["documents"]):
        metadata = peek_data["metadatas"][i]
        print("ID:", peek_data["ids"][i])
        print("Document:", doc)
        print("Answer:", metadata.get("answer"))
        print("—" * 40)

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
        query="Yêu mình đi mà",
        k=3,
        filter={"has_answer": True}
    )
    for doc, score in results:
        print(score, doc.page_content)

# render_data()
# retrieve_data()
similarity_search()



