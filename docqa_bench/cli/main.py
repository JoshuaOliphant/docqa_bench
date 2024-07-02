import asyncio
import click
import json
from openai import OpenAI
from docqa_bench.benchmark import Benchmark
from docqa_bench.document_types.pdf_document import PDFDocument
from docqa_bench.chunkers.simple_chunker import SimpleChunker
from docqa_bench.embedders.openai_embedder import OpenAIEmbedder
from docqa_bench.vector_stores.chroma_store import ChromaStore
from docqa_bench.models.openai_model import OpenAIQuestionGenerator, OpenAIAnswerGenerator
from docqa_bench.metrics.f1_score import F1Evaluator


@click.command()
@click.option('--pdf', type=click.Path(exists=True), help='Path to PDF file')
@click.option('--openai-key', help='OpenAI API Key')
@click.option('--chunk-size', default=1000, help='Size of text chunks')
@click.option('--chunk-overlap', default=200, help='Overlap between chunks')
@click.option('--embedding-model',
              default='text-embedding-ada-002',
              help='OpenAI embedding model')
def main(pdf: str, openai_key: str, chunk_size: int, chunk_overlap: int,
         embedding_model: str):
    asyncio.run(
        run_benchmark(pdf, openai_key, chunk_size, chunk_overlap,
                      embedding_model))


async def run_benchmark(pdf: str, openai_key: str, chunk_size: int,
                        chunk_overlap: int, embedding_model: str):
    OpenAI.api_key = openai_key  # Set the API key globally

    config = {
        'document':
        PDFDocument(pdf),
        'chunker':
        SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        'embedder':
        OpenAIEmbedder(model=embedding_model),
        'vector_store':
        ChromaStore('benchmark_collection'),
        'question_generator':
        OpenAIQuestionGenerator('gpt-4o'),
        'answer_generator':
        OpenAIAnswerGenerator('gpt-4o'),
        'evaluator':
        F1Evaluator()
    }

    benchmark = Benchmark(config)
    results = await benchmark.run()
    click.echo(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
