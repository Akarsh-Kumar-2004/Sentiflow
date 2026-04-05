from backend.model import SentimentModel


class StubPipeline:
    def __init__(self, responses):
        self.responses = responses

    def __call__(self, text):
        if isinstance(text, list):
            return [self.responses[item] for item in text]
        return [self.responses[text]]


def build_model(sentiment_responses, emotion_responses):
    model = SentimentModel.__new__(SentimentModel)
    model._sentiment_pipeline = StubPipeline(sentiment_responses)
    model._emotion_pipeline = StubPipeline(emotion_responses)
    model._sentiment_label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE",
    }
    model._emotion_label_map = {
        "joy": "HAPPY",
        "anger": "ANGRY",
        "fear": "FEAR",
        "sadness": "SAD",
        "surprise": "SURPRISE",
        "neutral": "NEUTRAL",
        "disgust": "ANGRY",
    }
    import re

    model._contrast_pattern = re.compile(
        r"\b(?:but|however|although|though|yet|while|whereas)\b",
        flags=re.IGNORECASE,
    )
    return model


def test_mixed_clause_text_is_not_forced_negative():
    text = "The phone camera is very good but battery is bad"
    sentiment_responses = {
        "The phone camera is very good": [
            {"label": "LABEL_2", "score": 0.85},
            {"label": "LABEL_1", "score": 0.10},
            {"label": "LABEL_0", "score": 0.05},
        ],
        "battery is bad": [
            {"label": "LABEL_0", "score": 0.82},
            {"label": "LABEL_1", "score": 0.12},
            {"label": "LABEL_2", "score": 0.06},
        ],
    }
    emotion_responses = {
        "The phone camera is very good": [
            {"label": "joy", "score": 0.90},
            {"label": "neutral", "score": 0.10},
        ],
        "battery is bad": [
            {"label": "anger", "score": 0.76},
            {"label": "sadness", "score": 0.24},
        ],
    }
    model = build_model(sentiment_responses, emotion_responses)

    result = model.predict(text)

    assert result["label"] == "MIXED"
    assert len(result["chunks"]) == 2
    assert {chunk["sentiment_label"] for chunk in result["chunks"]} == {"POSITIVE", "NEGATIVE"}


def test_split_text_detects_contrast_clauses():
    model = SentimentModel.__new__(SentimentModel)
    import re

    model._contrast_pattern = re.compile(
        r"\b(?:but|however|although|though|yet|while|whereas)\b",
        flags=re.IGNORECASE,
    )

    chunks = model._split_text("Great camera, but the battery is bad.")

    assert chunks == ["Great camera", "the battery is bad"]


def test_lowercase_model_labels_are_normalized():
    model = SentimentModel.__new__(SentimentModel)
    model._sentiment_label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE",
    }

    scores = model._normalize_scores(
        [
            {"label": "positive", "score": 0.97},
            {"label": "neutral", "score": 0.02},
            {"label": "negative", "score": 0.01},
        ]
    )

    assert scores["POSITIVE"] == 0.97
    assert scores["NEUTRAL"] == 0.02
    assert scores["NEGATIVE"] == 0.01
