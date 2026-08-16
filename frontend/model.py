from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

from transformers import pipeline


@dataclass
class ChunkAnalysis:
    text: str
    sentiment_label: str
    sentiment_score: float
    sentiment_value: float
    emotion_label: str
    emotion_score: float


class SentimentModel:
    def __init__(self):
        self._sentiment_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )
        self._emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        self._sentiment_label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE",
        }
        self._emotion_label_map = {
            "joy": "HAPPY",
            "anger": "ANGRY",
            "fear": "FEAR",
            "sadness": "SAD",
            "surprise": "SURPRISE",
            "neutral": "NEUTRAL",
            "disgust": "ANGRY",
        }
        self._contrast_pattern = re.compile(
            r"\b(?:but|however|although|though|yet|while|whereas)\b",
            flags=re.IGNORECASE,
        )

    def predict(self, text: str) -> dict[str, Any]:
        chunks = self._split_text(text)
        chunk_analyses = [self._analyze_chunk(chunk) for chunk in chunks]
        aggregate = self._aggregate_chunks(chunk_analyses)
        aggregate["chunks"] = [chunk.__dict__ for chunk in chunk_analyses]
        return aggregate

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.predict(text) for text in texts]

    def _split_text(self, text: str) -> list[str]:
        pieces = re.split(r"[.!?;\n]+", text)
        chunks: list[str] = []
        for piece in pieces:
            stripped = piece.strip()
            if not stripped:
                continue
            sub_parts = [p.strip(" ,") for p in self._contrast_pattern.split(stripped)]
            chunks.extend([part for part in sub_parts if part])
        return chunks or [text.strip()]

    def _normalize_scores(self, scores: list[dict[str, Any]]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for item in scores:
            raw_label = str(item["label"])
            mapped_label = self._sentiment_label_map.get(raw_label, raw_label).upper()
            normalized[mapped_label] = float(item["score"])
        return normalized

    def _normalize_emotions(self, scores: list[dict[str, Any]]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for item in scores:
            emotion = self._emotion_label_map.get(item["label"].lower(), item["label"].upper())
            normalized[emotion] = normalized.get(emotion, 0.0) + float(item["score"])
        return normalized

    def _analyze_chunk(self, text: str) -> ChunkAnalysis:
        sentiment_scores = self._normalize_scores(self._sentiment_pipeline(text)[0])
        pos = sentiment_scores.get("POSITIVE", 0.0)
        neg = sentiment_scores.get("NEGATIVE", 0.0)
        neu = sentiment_scores.get("NEUTRAL", 0.0)

        sentiment_value = pos - neg
        sentiment_label = "NEUTRAL"
        if sentiment_value > 0.2:
            sentiment_label = "POSITIVE"
        elif sentiment_value < -0.2:
            sentiment_label = "NEGATIVE"
        elif max(pos, neg) > 0.5 and abs(pos - neg) < 0.15:
            sentiment_label = "MIXED"
        elif neu < 0.2 and pos > 0.25 and neg > 0.25:
            sentiment_label = "MIXED"

        confidence = max(pos, neg, neu)
        if sentiment_label == "MIXED":
            confidence = min(1.0, abs(pos - neg) + 0.35)

        emotion_scores = self._normalize_emotions(self._emotion_pipeline(text)[0])
        emotion_label, emotion_score = max(
            emotion_scores.items(), key=lambda item: item[1], default=("NEUTRAL", 0.0)
        )

        return ChunkAnalysis(
            text=text,
            sentiment_label=sentiment_label,
            sentiment_score=confidence,
            sentiment_value=sentiment_value,
            emotion_label=emotion_label,
            emotion_score=emotion_score,
        )

    def _aggregate_chunks(self, chunks: list[ChunkAnalysis]) -> dict[str, Any]:
        sentiment_value = mean(chunk.sentiment_value for chunk in chunks)
        sentiment_score = mean(chunk.sentiment_score for chunk in chunks)

        if any(chunk.sentiment_label == "MIXED" for chunk in chunks):
            sentiment_label = "MIXED"
        elif sentiment_value > 0.2:
            sentiment_label = "POSITIVE"
        elif sentiment_value < -0.2:
            sentiment_label = "NEGATIVE"
        else:
            positive_chunks = sum(chunk.sentiment_value > 0.2 for chunk in chunks)
            negative_chunks = sum(chunk.sentiment_value < -0.2 for chunk in chunks)
            sentiment_label = "MIXED" if positive_chunks and negative_chunks else "NEUTRAL"

        emotion_totals: dict[str, float] = {}
        for chunk in chunks:
            emotion_totals[chunk.emotion_label] = (
                emotion_totals.get(chunk.emotion_label, 0.0) + chunk.emotion_score
            )
        emotion_label, emotion_score = max(
            emotion_totals.items(), key=lambda item: item[1], default=("NEUTRAL", 0.0)
        )

        return {
            "label": sentiment_label,
            "score": round(float(sentiment_score), 4),
            "sentiment_value": round(float(sentiment_value), 4),
            "emotion": emotion_label,
            "emotion_score": round(float(emotion_score / max(len(chunks), 1)), 4),
        }
