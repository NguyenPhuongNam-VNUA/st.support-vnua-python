from google import genai
from google.genai import types
import numpy as np
from numpy.linalg import norm
import os
from dotenv import load_dotenv

from collection import get_gemini_embedding_model, get_vectorstore, get_duplicate_questions_vectorstore

load_dotenv()

def getDataEmbedding():
    """
    Lấy dữ liệu từ vectorstore.
    """
    embedding_model = get_gemini_embedding_model(task_type='RETRIEVAL_DOCUMENT')
    vectorstore = get_vectorstore(embedding_model)

    # Lấy tất cả các tài liệu trong vectorstore
    documents = vectorstore.get()
    print("RAG vectorstore documents:")
    print(documents)

    # print("Kết hợp lấy dữ liệu từ vectorstore RAG:")
    # for doc in documents['embeddings']:
    #     print(f"Embedding values: {doc}")
    #     print(f"Embedding length: {len(doc)}")
    #     print(f"Norm of embedding: {norm(np.array(doc)):.6f}")  # Kiểm tra độ chuẩn hoá

    vectorstore_duplicate = get_duplicate_questions_vectorstore()
    duplicate_documents = vectorstore_duplicate.get()
    print("Duplicate questions vectorstore documents:")
    print(duplicate_documents)

def getGeminiEmbeddingModel():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=["Em muốn xin giấy xác nhận là sinh viên thì xin ở đâu ạ?"],
        config=types.EmbedContentConfig(
            task_type='RETRIEVAL_DOCUMENT',
            output_dimensionality=1536
        )
    )

    [embedding_obj] = result.embeddings
    embedding_values_np = np.array(embedding_obj.values)
    normed_embedding = embedding_values_np / norm(embedding_values_np)

    print(f"Embedding values: {normed_embedding}")
    print(f"Embedding toList: {normed_embedding.tolist()[:10]}")
    print(f"Normed embedding length: {len(normed_embedding)}")
    print(f"Norm of normed embedding: {norm(normed_embedding):.6f}")  # Should be very close to 1

def countInputTokens():
    """
    Đếm số lượng input token của một text.
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    text = (f"""Em muốn xin giấy xác nhận là sinh viên thì xin ở đâu ạ?
            Giấy xác nhận sinh viên lấy ở đâu? 
            Xác nhận sinh viên lấy kiểu gì?""")
    result = client.models.count_tokens(
        model="gemini-embedding-001",
        contents=[text]
    )
    print(result)

    return result

from langchain_chroma import Chroma
def test_semantic_similarity():
    model = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
    vectorstore = Chroma(
        embedding_function=model,
        collection_name="test_semantic_similarity",
        persist_directory=None,
        collection_metadata={"hnsw:space": "cosine"}
    )

    contents = [
        "Em muốn xin giấy xác nhận là sinh viên thì xin ở đâu ạ?\nGiấy xác nhận sinh viên lấy ở đâu? Xác nhận sinh viên lấy kiểu gì?",
        "Thời gian đăng ký môn học trực tuyến của học kỳ này là khi nào?",
        "Em bị ốm thì có cần giấy tờ gì để xin nghỉ học không?",
    ]

    vectorstore.add_texts(
        texts=contents,
        ids=["doc1", "doc2", "doc3"],
    )

    documents = vectorstore.get(include=['embeddings','documents'])
    ids = documents['ids']
    texts = documents["documents"]

    # print(range(len(ids)))
    # for i in range(len(ids)):
    #     print("====== Document ID:", ids[i], "======")
    #     print("Text:", texts[i][:100])

    #Demo Retrieval_Query
    model_query = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
    vector = model_query.embed_query("Muốn lấy giấy xác nhận sinh viên kiểu nào?")
    # print("Embedding vector:", vector[:10])  # In ra 10 giá trị đầu tiên của vector

    result = vectorstore.similarity_search_by_vector_with_relevance_scores(
        embedding=vector,
        k=2,
    )
    print("[✓] Kết quả tìm kiếm tương tự:")
    for doc, score in result:
        print(f"Document ID: {doc.id}, Score: {1 - score:.4f}")
        print(f"Content: {doc.page_content[:100]}...")

def is_duplicate_question(embed_data: str, threshold: float = 0.85):
    # Embedding câu hỏi để kiểm tra trùng lặp
    embedding_model = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
    vector = embedding_model.embed_query(embed_data)

    # Khởi tạo vectorstore duplicate để kiểm tra câu hỏi trùng lặp
    duplicate_vectorstore = get_duplicate_questions_vectorstore()
    results = duplicate_vectorstore.similarity_search_by_vector_with_relevance_scores(
        embedding=vector,
        k=1
    )
    print(1 - results[0][1])

def delete_embed(id: str):
    embedding_model = get_gemini_embedding_model(task_type='RETRIEVAL_DOCUMENT')
    vectorstore = get_vectorstore(embedding_model)
    vectorstore.delete(ids=[id])
    print(f"[✓] Xoá thành công id: {id}")

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# def test_langchain_google_genai():
#     embedding_model = get_gemini_embedding_model(task_type='RETRIEVAL_DOCUMENT')
#     vector = embedding_model.embed_query("Em muốn xin giấy xác nhận là sinh viên thì xin ở đâu ạ?")
#
#     embedding = GoogleGenerativeAIEmbeddings(
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#         model='gemini-embedding-001',
#         task_type='retrieval_document',
#     )
#
#     vector_lc = embedding.embed_query(
#         text="Em muốn xin giấy xác nhận là sinh viên thì xin ở đâu ạ?",
#         output_dimensionality=1536
#     )
#     print(vector[:10])
#     print(vector_lc[:10])
#     print(len(vector))
#     print(len(vector_lc))


# getDataEmbedding()
# getGeminiEmbeddingModel()
# countInputTokens()
# test_semantic_similarity()
# is_duplicate_question(f"Làm sao để có giấy xác nhận sinh viên nhỉ?\n")
# delete_embed("112")
# test_langchain_google_genai()