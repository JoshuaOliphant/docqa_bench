from typing import List, Dict, Any
from docqa_bench.core.document import BaseDocument, PreprocessedDocument
from docqa_bench.core.chunker import BaseChunker
from docqa_bench.core.embedder import BaseEmbedder
from docqa_bench.core.vector_store import BaseVectorStore
from docqa_bench.core.question_generator import BaseQuestionGenerator
from docqa_bench.core.answer_generator import BaseAnswerGenerator
from docqa_bench.core.evaluator import BaseEvaluator


class Benchmark:

    def __init__(self, document: BaseDocument, chunker: BaseChunker,
                 embedder: BaseEmbedder, vector_store: BaseVectorStore,
                 question_generator: BaseQuestionGenerator,
                 answer_generator: BaseAnswerGenerator,
                 evaluator: BaseEvaluator):
        self.document = document
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.question_generator = question_generator
        self.answer_generator = answer_generator
        self.evaluator = evaluator

    async def run(self) -> List[Dict[str, Any]]:
        content = await self.document.get_content()
        chunks = await self.chunker.chunk(content)

        # Use batch embedding
        embeddings = await self.embedder.embed_batch(chunks)

        # Filter out empty embeddings
        valid_chunks_and_embeddings = [
            (chunk, embedding) for chunk, embedding in zip(chunks, embeddings)
            if embedding
        ]

        print(f"Number of valid chunks: {len(valid_chunks_and_embeddings)}")

        # Add chunks to vector store
        for i, (chunk, embedding) in enumerate(valid_chunks_and_embeddings):
            try:
                await self.vector_store.add(f"chunk_{i}", embedding,
                                            {"text": chunk})
            except Exception as e:
                print(f"Error adding chunk to vector store: {e}")

        # Print the number of items in the vector store
        print(
            f"Number of items in vector store: {await self.vector_store.count()}"
        )

        questions = await self.question_generator.generate(content, n=10)

        results = []
        for question in questions:
            question_embedding = await self.embedder.embed(question)
            if not question_embedding:
                print(f"Failed to generate embedding for question: {question}")
                continue
            relevant_chunks = await self.vector_store.search(
                question_embedding, k=3)
            context = " ".join(
                [chunk['metadata']['text'] for chunk in relevant_chunks])
            generated_answer = await self.answer_generator.generate(
                question, context)
            reference_answer = await self.answer_generator.generate(
                question, content)
            score = await self.evaluator.evaluate(generated_answer,
                                                  reference_answer)
            results.append({
                "question": question,
                "generated_answer": generated_answer,
                "reference_answer": reference_answer,
                "score": score
            })

        return results

    @classmethod
    async def evaluate_scraped_content(
            cls, content: str, chunker: BaseChunker, embedder: BaseEmbedder,
            vector_store: BaseVectorStore,
            question_generator: BaseQuestionGenerator,
            answer_generator: BaseAnswerGenerator,
            evaluator: BaseEvaluator) -> Dict[str, Any]:
        document = PreprocessedDocument(content)
        benchmark = cls(document, chunker, embedder, vector_store,
                        question_generator, answer_generator, evaluator)
        results = await benchmark.run()

        return {
            "input_content": content,
            "results": results,
            "metadata": {
                "num_chunks": await vector_store.count(),
                # Add any other relevant metadata here
            }
        }
