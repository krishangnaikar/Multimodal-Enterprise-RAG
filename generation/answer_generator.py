from __future__ import annotations
from utils.ollama_client import ollama_chat

SYSTEM = "You are a precise enterprise assistant. Ground answers in the provided context."

ANSWER_PROMPT = """
Question: {q}

Use the CONTEXT to answer concisely. If unknown, say you don't know.
CONTEXT:
{context}
"""

def generate_answer(q, contexts, model = "mistral"):
    joined = "\n\n---\n".join([c.get("text", "") for c in contexts[:8]])
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ANSWER_PROMPT.format(q=q, context=joined[:8000])},
    ]
    return ollama_chat(model, messages)
