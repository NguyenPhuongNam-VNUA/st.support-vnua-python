from typing import List
import numpy as np
from numpy.linalg import norm
from google import genai
from google.genai import types
from app.core.config import settings

class CustomGeminiEmbeddings:
    """Service chịu trách nhiệm tính toán Vector Embedding và đếm Token sử dụng Google Gemini API."""

    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT"):
        self.task_type = task_type
        self.model = "gemini-embedding-001"
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=self.task_type,
                    output_dimensionality=1536
                )
            )
            embedding_obj = response.embeddings[0]
            embedding_value_np = np.array(embedding_obj.values)
            normed_embedding = (embedding_value_np / norm(embedding_value_np)).tolist()
            embeddings.append(normed_embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=1536
            )
        )
        embedding_obj = response.embeddings[0]
        embedding_value_np = np.array(embedding_obj.values)
        normed_embedding = (embedding_value_np / norm(embedding_value_np)).tolist()
        return normed_embedding

    @staticmethod
    def get_input_token_count(text: str) -> int:
        """Đếm số lượng input token của một văn bản."""
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        response = client.models.count_tokens(
            model="gemini-embedding-001",
            contents=[text]
        )
        return response.total_tokens
