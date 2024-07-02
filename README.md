# DocQA-Bench

DocQA-Bench is a tool for benchmarking documents with a question-answering system. It provides a flexible framework for evaluating various components of a QA pipeline, including document chunking, embedding generation, vector storage, question generation, and answer generation.

## Features

- Asynchronous processing for improved performance
- Modular design allowing easy swapping of components
- Support for PDF documents
- Integration with OpenAI's GPT models for question and answer generation
- Customizable chunk sizes and overlap for document processing
- F1 score evaluation for answer quality

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/docqa-bench.git
cd docqa-bench
2. Install the required dependencies using Poetry:
poetry install

## Usage

To run a benchmark, use the following command:
poetry run python -m docqa_bench.cli.main --pdf <path_to_pdf> --chunk-size <chunk_size> --chunk-overlap <chunk_overlap> --embedding-model <embedding_model>

For example:
poetry run python -m docqa_bench.cli.main --pdf path/to/pdf --chunk-size 1000 --chunk-overlap 200 --embedding-model text-embedding-ada-002

Make sure to set your OpenAI API key as an environment variable:
export OPENAI_API_KEY=your_api_key_here

## Project Structure

- `docqa_bench/`: Main package directory
  - `core/`: Core abstract base classes
  - `document_types/`: Document handling (e.g., PDFDocument)
  - `chunkers/`: Text chunking implementations
  - `embedders/`: Embedding generation implementations
  - `vector_stores/`: Vector storage implementations
  - `models/`: Question and answer generation models
  - `metrics/`: Evaluation metrics
  - `cli/`: Command-line interface
  - `benchmark.py`: Main benchmarking logic

## Extending the Project

To add new components:

1. Implement the relevant abstract base class from the `core/` directory
2. Add your implementation to the appropriate subdirectory
3. Update the `cli/main.py` file to include your new component as an option

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
