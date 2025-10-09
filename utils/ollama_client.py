from __future__ import annotations
import os
import json
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _parse_ollama_response_text(txt):
    s = txt.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    last_obj = None
    for line in s.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            last_obj = json.loads(line)
        except Exception:
            continue
    if last_obj is not None:
        return last_obj

    return {}


def ollama_chat(model, messages, temperature = 0.2):
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False,
    }
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=600,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()

    data = None
    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        data = _parse_ollama_response_text(resp.text)

    if not isinstance(data, dict):
        return ""

    msg = data.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content

    if isinstance(data.get("content"), str):
        return data["content"]

    if isinstance(data.get("messages"), list) and data["messages"]:
        for m in reversed(data["messages"]):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return m["content"]

    return ""


def ollama_embed(model, text_list):
    vectors = []
    for t in text_list:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=600,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            data = _parse_ollama_response_text(resp.text)

        emb = data.get("embedding")
        if isinstance(emb, list):
            vectors.append(emb)
        else:
            vectors.append([])
    return vectors
