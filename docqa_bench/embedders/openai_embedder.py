import asyncio
from typing import List
from openai import OpenAI
from docqa_bench.core.embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):

    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        self.client = OpenAI()

    async def embed(self, text: str) -> List[float]:
        try:
            response = await asyncio.to_thread(self.client.embeddings.create,
                                               model=self.model,
                                               input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in OpenAIEmbedder: {str(e)}")
            return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await asyncio.to_thread(self.client.embeddings.create,
                                               model=self.model,
                                               input=texts)
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error in OpenAIEmbedder batch embedding: {str(e)}")
            return [[] for _ in texts]
