from __future__ import annotations
import json, re
from utils.ollama_client import ollama_chat

SYSTEM = "You extract entities and relationships from text into JSON triplets."

PROMPT = """Extract entities and relationships from the text as triplets.
Return JSON with this schema:
{{
  "entities": [{{"id": "<string>", "label": "<type>", "name": "<string>"}}],
  "relations": [{{"source": "<entity_id>", "type": "<string>", "target": "<entity_id>", "evidence": "<short quote>"}}]
}}

Constraints:
- Return ONLY valid JSON (no commentary, no markdown fences).
- Use short, stable ids for entities (e.g., "e1", "e2", ...).
- If nothing is found, return: {{"entities": [], "relations": []}}.

Text:
\"\"\"{text}\"\"\""""

def _extract_json_block(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, count=1)
        s = re.sub(r"\s*```$", "", s, count=1)

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else s

def extract_triples(text, model = "mistral"):
    clipped = text if text else ""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(text=clipped)},
    ]
    out = ollama_chat(model, messages)

    try:
        payload = _extract_json_block(out)
        data = json.loads(payload)
        entities = data.get("entities", []) if isinstance(data, dict) else []
        relations = data.get("relations", []) if isinstance(data, dict) else []
        if not isinstance(entities, list): entities = []
        if not isinstance(relations, list): relations = []
        return entities, relations
    except Exception:
        return [], []
