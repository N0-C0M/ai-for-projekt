#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaModel:
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 60) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.host}{path}"

    def list_models(self) -> List[OllamaModel]:
        try:
            resp = self._session.get(self._url("/api/tags"), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise OllamaError(f"Не удалось подключиться к Ollama: {exc}") from exc
        except ValueError as exc:
            raise OllamaError(f"Некорректный ответ Ollama: {exc}") from exc

        models: List[OllamaModel] = []
        for item in data.get("models", []) or []:
            models.append(
                OllamaModel(
                    name=item.get("name", ""),
                    size=item.get("size"),
                    modified_at=item.get("modified_at"),
                    digest=item.get("digest"),
                    details=item.get("details"),
                )
            )
        return models

    def chat(self, messages: List[Dict[str, str]], model: str, stream: bool = False) -> str:
        payload = {"model": model, "messages": messages, "stream": stream}
        try:
            resp = self._session.post(
                self._url("/api/chat"), json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise OllamaError(f"Запрос к Ollama не удался: {exc}") from exc
        except ValueError as exc:
            raise OllamaError(f"Некорректный JSON от Ollama: {exc}") from exc

        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise OllamaError("Ollama вернула пустой ответ.")
        return content
