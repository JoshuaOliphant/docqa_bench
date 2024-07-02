import aiofiles
from pypdf import PdfReader
from io import BytesIO
from docqa_bench.core.document import BaseDocument


class PDFDocument(BaseDocument):

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def get_content(self) -> str:
        async with aiofiles.open(self.file_path, 'rb') as f:
            pdf_content = await f.read()

        reader = PdfReader(BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    async def get_metadata(self) -> dict:
        async with aiofiles.open(self.file_path, 'rb') as f:
            pdf_content = await f.read()

        reader = PdfReader(BytesIO(pdf_content))
        return dict(reader.metadata)
