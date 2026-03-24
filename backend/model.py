from transformers import pipeline


class SentimentModel:
    def __init__(self):
        # distilbert-base-uncased-finetuned-sst-2-english is a small sentiment model
        self._pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def predict(self, text: str):
        results = self._pipeline(text)
        # results is a list like [{label: 'POSITIVE', score: 0.999}]
        return results[0]

    def predict_batch(self, texts: list[str]):
        results = self._pipeline(texts, batch_size=16)
        return results
