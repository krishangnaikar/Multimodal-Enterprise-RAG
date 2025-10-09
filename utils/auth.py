from __future__ import annotations
from pathlib import Path
import json, os, hmac, hashlib, base64

USERS_PATH = Path("state") / "users.json"
USERS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- password hashing (PBKDF2-HMAC-SHA256) ----
def _hash_password(password, salt, iters = 200_000):
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)
    return base64.b64encode(dk).decode("utf-8")

def _new_salt(n = 16):
    return os.urandom(n)

def load_users():
    if USERS_PATH.exists():
        try:
            return json.loads(USERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_users(users):
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")

def register_user(username, password):
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Username and password are required."
    users = load_users()
    if username in users:
        return False, "Username already exists."
    salt = _new_salt()
    pwd_hash = _hash_password(password, salt)
    users[username] = {
        "salt": base64.b64encode(salt).decode("utf-8"),
        "hash": pwd_hash,
        "iters": 200_000,
    }
    save_users(users)
    return True, "Registration successful."

def authenticate_user(username, password):
    username = (username or "").strip().lower()
    users = load_users()
    rec = users.get(username)
    if not rec:
        return False, "Invalid username or password."
    salt = base64.b64decode(rec["salt"])
    iters = int(rec.get("iters", 200_000))
    cand = _hash_password(password, salt, iters)
    if hmac.compare_digest(cand, rec["hash"]):
        return True, "Login successful."
    return False, "Invalid username or password."
