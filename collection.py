import os
from langchain_chroma import Chroma
from gemini_embedding_001.CustomGeminiEmbeddings import CustomGeminiEmbeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_gemini_embedding_model(task_type: str='RETRIEVAL_DOCUMENT'):
    """Khởi tạo embedding model với task_type cụ thể."""
    return CustomGeminiEmbeddings(task_type=task_type)

def get_vectorstore(embedding_model):
    """Lấy đối tượng vectorstore đã được khởi tạo."""
    return Chroma(
        collection_name="qa_rag_collection",
        embedding_function=embedding_model,
        persist_directory=DATA_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

def get_duplicate_questions_vectorstore():
    """Lấy vectorstore cho các câu hỏi trùng lặp."""
    embedding_model = get_gemini_embedding_model(task_type='SEMANTIC_SIMILARITY')
    return Chroma(
        collection_name="duplicate_questions",
        embedding_function=embedding_model,
        persist_directory=DATA_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
