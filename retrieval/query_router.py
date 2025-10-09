from __future__ import annotations
import json, re
from utils.ollama_client import ollama_chat

SYSTEM = "You are a query triage router. Decide intent."

PROMPT = """Classify the user query into one of:
[lookup, summarization, reasoning, graph_explore]

Return JSON exactly in this shape:
{{
  "intent": "<one_of_lookup|summarization|reasoning|graph_explore>",
  "rewrite": "<short better query>"
}}

Constraints:
- Return ONLY valid JSON (no prose, no code fences).
- If unsure, use intent="lookup" and rewrite=the original query.

Query: "{q}"
"""

def _extract_json_block(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, count=1)
        s = re.sub(r"\s*```$", "", s, count=1)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else s

def route_query(q, model = "mistral"):
    try:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(q=q)},
        ]
        raw = ollama_chat(model, messages)
        payload = _extract_json_block(raw)
        data = json.loads(payload) if payload else {}
        intent = data.get("intent", "lookup")
        rewrite = data.get("rewrite", q)
        if not isinstance(intent, str) or not intent:
            intent = "lookup"
        if not isinstance(rewrite, str) or not rewrite:
            rewrite = q
        return {"intent": intent, "rewrite": rewrite}
    except Exception:
        return {"intent": "lookup", "rewrite": q}
