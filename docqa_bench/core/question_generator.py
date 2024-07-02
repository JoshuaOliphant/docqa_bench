from abc import ABC, abstractmethod
from typing import List

class BaseQuestionGenerator(ABC):
    @abstractmethod
    async def generate(self, context: str, n: int) -> List[str]:
        pass