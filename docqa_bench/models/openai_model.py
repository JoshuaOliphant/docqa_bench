import asyncio
from typing import List
from openai import OpenAI
from docqa_bench.core.question_generator import BaseQuestionGenerator
from docqa_bench.core.answer_generator import BaseAnswerGenerator


class OpenAIQuestionGenerator(BaseQuestionGenerator):

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI()

    async def generate(self, context: str, n: int) -> List[str]:
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{
                    "role":
                    "system",
                    "content":
                    "Generate questions based on the given context."
                }, {
                    "role":
                    "user",
                    "content":
                    f"Context: {context}\n\nGenerate {n} questions:"
                }])
            return response.choices[0].message.content.strip().split('\n')
        except Exception as e:
            print(f"Error in OpenAIQuestionGenerator: {str(e)}")
            return []


class OpenAIAnswerGenerator(BaseAnswerGenerator):

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI()

    async def generate(self, question: str, context: str) -> str:
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{
                    "role":
                    "system",
                    "content":
                    "Answer the question based on the given context."
                }, {
                    "role":
                    "user",
                    "content":
                    f"Context: {context}\n\nQuestion: {question}"
                }])
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in OpenAIAnswerGenerator: {str(e)}")
            return ""
