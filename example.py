import asyncio
import json
import argparse
from docqa_bench import (
    Benchmark,
    PreprocessedDocument,
    SimpleChunker,
    OpenAIEmbedder,
    ChromaStore,
    OpenAIQuestionGenerator,
    OpenAIAnswerGenerator,
    F1Evaluator,
)


async def run_benchmark(content: str):
    # Create components
    document = PreprocessedDocument(content)
    chunker = SimpleChunker(chunk_size=1000, chunk_overlap=200)
    embedder = OpenAIEmbedder("text-embedding-ada-002")
    vector_store = ChromaStore("benchmark_collection")
    question_generator = OpenAIQuestionGenerator("gpt-4o-mini")
    answer_generator = OpenAIAnswerGenerator("gpt-4o-mini")
    evaluator = F1Evaluator()

    # Create and run benchmark
    benchmark = Benchmark(
        document,
        chunker,
        embedder,
        vector_store,
        question_generator,
        answer_generator,
        evaluator,
    )
    results = await benchmark.run()

    # Prepare detailed results
    detailed_results = []
    for result in results:
        detailed_result = {
            "question": result["question"],
            "generated_answer": result["generated_answer"],
            "reference_answer": result["reference_answer"],
            "score": result["score"],
        }
        detailed_results.append(detailed_result)

    return detailed_results


async def main():
    parser = argparse.ArgumentParser(description="Run docqa_bench example")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    # Example preprocessed content
    content = """
    The Python programming language was created by Guido van Rossum and first released in 1991. 
    Python is known for its simplicity and readability, making it an excellent choice for beginners and experts alike. 
    It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.
    """

    results = await run_benchmark(content)

    # Prepare output
    output = {"input_content": content, "results": results}

    # Convert to JSON
    json_output = json.dumps(output, indent=2)

    # Print to console
    print(json_output)

    # Save to file if specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
