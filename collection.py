from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma(
    collection_name="qa_rag_collection",
    embedding_function=embedding,
    persist_directory=data_dir,
    collection_metadata={"hnsw:space": "cosine"}
)
