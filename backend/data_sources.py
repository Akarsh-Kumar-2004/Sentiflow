from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests


class TopicDataError(RuntimeError):
    pass


class TopicDataService:
    def fetch(self, keyword: str, source: str = "all", limit: int = 50) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        normalized_source = source.lower()

        if normalized_source in {"all", "newsapi"}:
            items.extend(self._fetch_newsapi(keyword, limit))
        if normalized_source in {"all", "gnews"}:
            items.extend(self._fetch_gnews(keyword, limit))
        if normalized_source in {"all", "twitter"}:
            items.extend(self._fetch_twitter(keyword, limit))

        deduped: dict[str, dict[str, Any]] = {}
        for item in items:
            text_key = item["text"].strip().lower()
            if text_key:
                deduped[text_key] = item

        sorted_items = sorted(
            deduped.values(),
            key=lambda item: item.get("published_at") or "",
            reverse=True,
        )
        return sorted_items[:limit]

    def _fetch_newsapi(self, keyword: str, limit: int) -> list[dict[str, Any]]:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return []

        page_size = min(limit, 100)
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": keyword,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])

        return [
            {
                "source": "NewsAPI",
                "author": article.get("source", {}).get("name"),
                "title": article.get("title"),
                "text": self._join_text(article.get("title"), article.get("description")),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
            }
            for article in articles
            if self._join_text(article.get("title"), article.get("description"))
        ]

    def _fetch_gnews(self, keyword: str, limit: int) -> list[dict[str, Any]]:
        api_key = os.getenv("GNEWS_API_KEY")
        if not api_key:
            return []

        page_size = min(limit, 100)
        response = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": keyword,
                "lang": "en",
                "sortby": "publishedAt",
                "max": page_size,
                "token": api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])

        return [
            {
                "source": "GNews",
                "author": article.get("source", {}).get("name"),
                "title": article.get("title"),
                "text": self._join_text(article.get("title"), article.get("description")),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
            }
            for article in articles
            if self._join_text(article.get("title"), article.get("description"))
        ]

    def _fetch_twitter(self, keyword: str, limit: int) -> list[dict[str, Any]]:
        try:
            import snscrape.modules.twitter as sntwitter
        except ImportError:
            return []

        items: list[dict[str, Any]] = []
        for index, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
            if index >= limit:
                break
            text = getattr(tweet, "rawContent", None) or getattr(tweet, "content", "")
            if not text:
                continue
            items.append(
                {
                    "source": "Twitter",
                    "author": getattr(tweet.user, "username", None) if getattr(tweet, "user", None) else None,
                    "title": None,
                    "text": text,
                    "url": getattr(tweet, "url", None),
                    "published_at": self._to_iso(getattr(tweet, "date", None)),
                }
            )
        return items

    def _join_text(self, *parts: Any) -> str:
        cleaned = [str(part).strip() for part in parts if part and str(part).strip()]
        return " ".join(cleaned)

    def _to_iso(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
