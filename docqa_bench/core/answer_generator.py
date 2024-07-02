from abc import ABC, abstractmethod

class BaseAnswerGenerator(ABC):
    @abstractmethod
    async def generate(self, question: str, context: str) -> str:
        pass