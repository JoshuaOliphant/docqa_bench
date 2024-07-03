from abc import ABC, abstractmethod
from typing import List, Dict


class BaseVectorStore(ABC):

    @abstractmethod
    async def add(self, id: str, vector: List[float], metadata: dict):
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], k: int) -> List[Dict]:
        pass

    @abstractmethod
    async def count(self) -> int:
        pass
