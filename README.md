# DocQA-Bench

DocQA-Bench is a library for benchmarking document question-answering systems. It provides a flexible framework for evaluating various components of a QA pipeline, including document chunking, embedding generation, vector storage, question generation, and answer generation.

## Features

- Modular design allowing easy swapping of components
- Support for custom document preprocessing
- Integration with OpenAI's GPT models for question and answer generation
- Customizable chunking strategies
- Vector store integration for efficient similarity search
- F1 score evaluation for answer quality
- Asynchronous processing for improved performance

## Installation

You can install DocQA-Bench using pip:

```bash
pip install docqa-bench
```

## Quick Start
Check out the example.py file in the root directory for a demonstration of how to use docqa_bench. This example shows how to:

Set up the necessary components
Create a benchmark
Run the benchmark on preprocessed content
Handle and display the results

To run the example:
`python example.py`

## Usage

Here's a basic example of how to use DocQA-Bench:

```python
import asyncio
from docqa_bench import (
    Benchmark, PreprocessedDocument, SimpleChunker, OpenAIEmbedder,
    ChromaStore, OpenAIQuestionGenerator, OpenAIAnswerGenerator, F1Evaluator
)

async def run_benchmark(content: str):
    document = PreprocessedDocument(content)
    chunker = SimpleChunker(chunk_size=1000, chunk_overlap=200)
    embedder = OpenAIEmbedder("text-embedding-ada-002")
    vector_store = ChromaStore("benchmark_collection")
    question_generator = OpenAIQuestionGenerator("gpt-3.5-turbo")
    answer_generator = OpenAIAnswerGenerator("gpt-3.5-turbo")
    evaluator = F1Evaluator()

    benchmark = Benchmark(
        document, chunker, embedder, vector_store,
        question_generator, answer_generator, evaluator
    )
    
    results = await benchmark.run()
    print(results)

asyncio.run(run_benchmark("Your document content here"))
```

## Components

DocQA-Bench consists of several modular components:

Document: Represents the input document. Use PreprocessedDocument for already processed text.
Chunker: Splits the document into manageable pieces. SimpleChunker is provided out of the box.
Embedder: Generates vector representations of text. OpenAIEmbedder uses OpenAI's embedding model.
VectorStore: Stores and retrieves vector representations. ChromaStore is provided as a default implementation.
QuestionGenerator: Generates questions based on the document content. OpenAIQuestionGenerator uses OpenAI's GPT model.
AnswerGenerator: Generates answers to questions. OpenAIAnswerGenerator uses OpenAI's GPT model.
Evaluator: Evaluates the quality of generated answers. F1Evaluator is provided as a default implementation.

## Customization

You can create custom implementations of any component by subclassing the respective base classes in the docqa_bench.core module.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

##Acknowledgments

OpenAI for their GPT and embedding models
Chroma for their vector database

## Support

If you encounter any problems or have any questions, please open an issue on the GitHub repository.