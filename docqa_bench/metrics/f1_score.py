from docqa_bench.core.evaluator import BaseEvaluator
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


class F1Evaluator(BaseEvaluator):

    async def evaluate(self, generated_answer: str,
                       reference_answer: str) -> float:
        # Convert answers to sets of words
        generated_words = set(generated_answer.lower().split())
        reference_words = set(reference_answer.lower().split())

        # Create a vocabulary of all unique words
        vocab = list(generated_words.union(reference_words))

        # Create binary vectors
        mlb = MultiLabelBinarizer(classes=vocab)
        y_true = mlb.fit_transform([reference_words])
        y_pred = mlb.transform([generated_words])

        # Calculate F1 score
        return f1_score(y_true, y_pred, average='micro')
