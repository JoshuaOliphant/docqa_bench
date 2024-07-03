import asyncio
import logging
from typing import List, Dict
import chromadb
from docqa_bench.core.vector_store import BaseVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):

    def __init__(self, collection_name: str):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    async def add(self, id: str, vector: List[float], metadata: dict):
        try:
            await asyncio.to_thread(self.collection.add,
                                    embeddings=[vector],
                                    documents=[metadata.get('text', '')],
                                    metadatas=[metadata],
                                    ids=[id])
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
            raise

    async def search(self, query_vector: List[float], k: int) -> List[Dict]:
        try:
            results = await asyncio.to_thread(self.collection.query,
                                              query_embeddings=[query_vector],
                                              n_results=k)

            if not results or not results.get('ids'):
                logger.warning("No results found in ChromaDB search")
                return []

            return [{
                "id": id,
                "score": score,
                "metadata": metadata
            } for id, score, metadata in
                    zip(results['ids'][0], results['distances'][0],
                        results['metadatas'][0])]
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    async def count(self) -> int:
        try:
            return await asyncio.to_thread(self.collection.count)
        except Exception as e:
            logger.error(f"Error counting ChromaDB entries: {e}")
            return 0
