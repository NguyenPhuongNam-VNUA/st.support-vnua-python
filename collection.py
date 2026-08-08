import os
import chromadb
from gemini_embedding_001.CustomGeminiEmbeddings import CustomGeminiEmbeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_chroma_client = chromadb.PersistentClient(path=DATA_DIR)

def get_gemini_embedding_model(task_type: str = 'RETRIEVAL_DOCUMENT'):
    """Khởi tạo embedding model với task_type cụ thể."""
    return CustomGeminiEmbeddings(task_type=task_type)

def get_vectorstore():
    """Lấy collection qa_rag_collection từ ChromaDB."""
    return _chroma_client.get_or_create_collection(
        name="qa_rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

def get_duplicate_questions_vectorstore():
    """Lấy collection duplicate_questions từ ChromaDB."""
    return _chroma_client.get_or_create_collection(
        name="duplicate_questions",
        metadata={"hnsw:space": "cosine"}
    )

