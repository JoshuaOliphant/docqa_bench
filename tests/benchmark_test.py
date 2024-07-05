import pytest
import uuid
from docqa_bench import (Benchmark, PreprocessedDocument, SimpleChunker,
                         OpenAIEmbedder, ChromaStore, OpenAIQuestionGenerator,
                         OpenAIAnswerGenerator, F1Evaluator)

# Mock data
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. This is a sample text for testing purposes."
SAMPLE_QUESTION = "What does the fox do?"
SAMPLE_ANSWER = "The fox jumps over the lazy dog."


# Fixtures
@pytest.fixture
def document():
    return PreprocessedDocument(SAMPLE_TEXT)


@pytest.fixture
def chunker():
    return SimpleChunker(chunk_size=20, chunk_overlap=5)


@pytest.fixture
def embedder():
    return OpenAIEmbedder("text-embedding-ada-002")


@pytest.fixture
def vector_store():
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    return ChromaStore(collection_name)


@pytest.fixture
def question_generator():
    return OpenAIQuestionGenerator("gpt-3.5-turbo-0125")


@pytest.fixture
def answer_generator():
    return OpenAIAnswerGenerator("gpt-3.5-turbo-0125")


@pytest.fixture
def evaluator():
    return F1Evaluator()


@pytest.fixture
def benchmark(document, chunker, embedder, vector_store, question_generator,
              answer_generator, evaluator):
    return Benchmark(document, chunker, embedder, vector_store,
                     question_generator, answer_generator, evaluator)


# Tests
@pytest.mark.asyncio
async def test_preprocessed_document(document):
    content = await document.get_content()
    assert content == SAMPLE_TEXT


@pytest.mark.asyncio
async def test_simple_chunker(chunker):
    chunks = await chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)


@pytest.mark.asyncio
async def test_openai_embedder(embedder):
    embedding = await embedder.embed(SAMPLE_TEXT)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_chroma_store(vector_store):
    embedding = [0.1] * 1536  # Assuming 1536-dimensional embeddings
    await vector_store.add("test_id", embedding, {"text": SAMPLE_TEXT})
    results = await vector_store.search(embedding, k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["text"] == SAMPLE_TEXT


@pytest.mark.asyncio
async def test_openai_question_generator(question_generator):
    questions = await question_generator.generate(SAMPLE_TEXT, n=1)
    print(f"Generated questions: {questions}")
    assert len(questions) == 1
    assert isinstance(questions[0], str)


@pytest.mark.asyncio
async def test_openai_answer_generator(answer_generator):
    answer = await answer_generator.generate(SAMPLE_QUESTION, SAMPLE_TEXT)
    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_f1_evaluator(evaluator):
    score = await evaluator.evaluate(SAMPLE_ANSWER, SAMPLE_ANSWER)
    assert score == 1.0  # Perfect match should give a score of 1.0

    score = await evaluator.evaluate("The fox jumps", SAMPLE_ANSWER)
    assert 0 < score < 1  # Partial match should give a score between 0 and 1


@pytest.mark.asyncio
async def test_benchmark_run(benchmark):
    results = await benchmark.run()
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(result, dict) for result in results)
    assert all("question" in result and "score" in result
               for result in results)


@pytest.mark.asyncio
async def test_benchmark_evaluate_scraped_content():
    results = await Benchmark.evaluate_scraped_content(
        SAMPLE_TEXT, SimpleChunker(chunk_size=20, chunk_overlap=5),
        OpenAIEmbedder("text-embedding-ada-002"),
        ChromaStore("test_collection"),
        OpenAIQuestionGenerator("gpt-3.5-turbo"),
        OpenAIAnswerGenerator("gpt-3.5-turbo"), F1Evaluator())
    assert isinstance(results, dict)
    assert "input_content" in results
    assert "results" in results
    assert "metadata" in results


# Integration test
@pytest.mark.asyncio
async def test_full_pipeline(benchmark):
    results = await benchmark.run()
    assert len(results) > 0
    for result in results:
        assert "question" in result
        assert "generated_answer" in result
        assert "reference_answer" in result
        assert "score" in result
        assert 0 <= result["score"] <= 1
