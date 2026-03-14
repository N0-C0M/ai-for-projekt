#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from ollama_handler import OllamaClient, OllamaError


def chat(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> str:
    provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()

    if provider == "ollama":
        host = base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        model = model or os.getenv("OLLAMA_MODEL") or "llama3"
        client = OllamaClient(host=host, timeout=timeout)
        try:
            return client.chat(messages, model=model, stream=False)
        except OllamaError as exc:
            raise RuntimeError(str(exc)) from exc

    if provider == "openai":
        host = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
        model = model or os.getenv("LLM_MODEL") or "local-model"
        url = f"{host.rstrip('/')}/chat/completions"
        headers = {}
        api_key = api_key or os.getenv("LLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise ValueError("Неизвестный провайдер. Используйте 'ollama' или 'openai'.")


def build_summary_messages(context: Dict[str, Any], question: Optional[str] = None):
    system_msg = (
        "Ты помощник по анализу данных. Отвечай на русском. "
        "Будь кратким, практичным и упоминай ключевые риски или ограничения."
    )
    user_msg = (
        "Сформулируй краткую сводку результатов обучения модели для прогноза "
        "посещаемости культурных событий. Используй контекст ниже и дай 5–8 пунктов."
    )
    if question:
        user_msg += f"\n\nВопрос пользователя: {question}"

    user_msg += "\n\nКонтекст:\n" + _safe_json_dumps(context)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_qa_messages(context: Dict[str, Any], question: str):
    system_msg = (
        "Ты помощник по анализу данных. Отвечай на русском. "
        "Используй предоставленный контекст и отмечай неопределенности."
    )
    user_msg = f"Вопрос: {question}\n\nКонтекст:\n" + _safe_json_dumps(context)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    return str(obj)
