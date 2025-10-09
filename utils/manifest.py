from __future__ import annotations
from pathlib import Path
import json, hashlib

def _user_dir(user):
    d = Path("state") / "users" / user
    d.mkdir(parents=True, exist_ok=True)
    return d

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest(user):
    p = _user_dir(user) / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(user, data):
    p = _user_dir(user) / "manifest.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
