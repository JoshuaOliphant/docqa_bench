from abc import ABC, abstractmethod

class BaseDocument(ABC):
    @abstractmethod
    async def get_content(self) -> str:
        pass

    @abstractmethod
    async def get_metadata(self) -> dict:
        pass