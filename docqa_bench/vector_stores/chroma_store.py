import asyncio
from typing import List, Dict
import chromadb
from docqa_bench.core.vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):

    def __init__(self, collection_name: str):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    async def add(self, id: str, vector: List[float], metadata: dict):
        await asyncio.to_thread(self.collection.add,
                                embeddings=[vector],
                                documents=[metadata.get('text', '')],
                                metadatas=[metadata],
                                ids=[id])

    async def search(self, query_vector: List[float], k: int) -> List[Dict]:
        results = await asyncio.to_thread(self.collection.query,
                                          query_embeddings=[query_vector],
                                          n_results=k)
        return [{
            "id": id,
            "score": score,
            "metadata": metadata
        } for id, score, metadata in
                zip(results['ids'][0], results['distances'][0],
                    results['metadatas'][0])]
