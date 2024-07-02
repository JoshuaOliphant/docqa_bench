from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    async def chunk(self, text: str) -> List[str]:
        pass