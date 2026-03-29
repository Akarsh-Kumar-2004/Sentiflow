from transformers import pipeline

class SentimentModel:
    def __init__(self):
        self._pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )

        self.label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }

    def predict(self, text: str):
        res = self._pipeline(text)[0]
        return {
            "label": self.label_map.get(res["label"], res["label"]),
            "score": res["score"]
        }

    def predict_batch(self, texts: list[str]):
        results = self._pipeline(texts)
        return [
            {
                "label": self.label_map.get(r["label"], r["label"]),
                "score": r["score"]
            }
            for r in results
        ]