from typing import Dict, Any


class Benchmark:

    def __init__(self, config: Dict[str, Any]):
        self.document = config['document']
        self.chunker = config['chunker']
        self.embedder = config['embedder']
        self.vector_store = config['vector_store']
        self.question_generator = config['question_generator']
        self.answer_generator = config['answer_generator']
        self.evaluator = config['evaluator']

    async def run(self):
        content = await self.document.get_content()
        chunks = await self.chunker.chunk(content)

        # Use batch embedding
        embeddings = await self.embedder.embed_batch(chunks)

        # Filter out empty embeddings
        valid_chunks_and_embeddings = [
            (chunk, embedding) for chunk, embedding in zip(chunks, embeddings)
            if embedding
        ]

        # Add chunks to vector store
        for i, (chunk, embedding) in enumerate(valid_chunks_and_embeddings):
            try:
                await self.vector_store.add(f"chunk_{i}", embedding,
                                            {"text": chunk})
            except Exception as e:
                print(f"Error adding chunk to vector store: {e}")

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
            results.append({"question": question, "score": score})

        return results
