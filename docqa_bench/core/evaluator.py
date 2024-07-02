from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, generated_answer: str, reference_answer: str) -> float:
        pass