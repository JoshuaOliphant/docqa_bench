from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docqa_bench.core.chunker import BaseChunker

class SimpleChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    async def chunk(self, text: str) -> List[str]:
        # LangChain's text splitter is not async, so we'll run it in a separate thread
        return await self._split_text(text)

    async def _split_text(self, text: str) -> List[str]:
        # This method runs the synchronous text_splitter in a separate thread
        import asyncio
        return await asyncio.to_thread(self.text_splitter.split_text, text)