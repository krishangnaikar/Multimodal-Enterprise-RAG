from __future__ import annotations
from pathlib import Path
import json

STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

def _user_dir(username):
    d = STATE_DIR / "users" / username
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_file_index(username):
    reg_path = _user_dir(username) / "files_index.json"
    if reg_path.exists():
        try:
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def save_file_index(username, index):
    reg_path = _user_dir(username) / "files_index.json"
    reg_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
