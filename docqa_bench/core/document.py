from abc import ABC, abstractmethod


class BaseDocument(ABC):

    @abstractmethod
    async def get_content(self) -> str:
        pass


class PreprocessedDocument(BaseDocument):

    def __init__(self, content: str):
        self._content = content

    async def get_content(self) -> str:
        return self._content
