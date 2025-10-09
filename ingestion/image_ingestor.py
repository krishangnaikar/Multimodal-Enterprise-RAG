from __future__ import annotations
from pathlib import Path
from PIL import Image
import os, shutil
import base64
try:
    import pytesseract
except Exception:
    pytesseract = None

from utils.ollama_client import ollama_chat

def _encode_image_b64(path, max_side = 1280):
    p = Path(path)
    mime = "image/png" if p.suffix.lower() in {".png"} else "image/jpeg"

    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
        if scale > 1.0:
            img = img.resize((int(w/scale), int(h/scale)))
        import io
        buf = io.BytesIO()
        if mime == "image/png":
            img.save(buf, format="PNG")
        else:
            img.save(buf, format="JPEG", quality=88)
        raw = buf.getvalue()
    except Exception:
        raw = Path(path).read_bytes()

    return base64.b64encode(raw).decode("utf-8"), mime
def _detect_tesseract_path():
    if pytesseract is None:
        return None

    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
    if cmd and os.path.exists(cmd):
        return cmd

    win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(win_default):
        pytesseract.pytesseract.tesseract_cmd = win_default
        return win_default

    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        return which

    return None

_TESS_CMD = _detect_tesseract_path()
_HAVE_TESS = _TESS_CMD is not None

def ocr_image(path):
    if not _HAVE_TESS or pytesseract is None:
        return ""
    try:
        img = Image.open(path).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def caption_image_with_llava(path, model="llava"):
    try:
        b64, mime = _encode_image_b64(path)

        messages = [
            {"role": "system",
             "content": "You are a concise, factual vision captioner. Describe the key objects, text, and scene in 1–2 sentences."},
            {"role": "user",
             "content": "Please caption this image.",
             "images": [b64]},
        ]
        out = (ollama_chat(model, messages) or "").strip()
        if out:
            return out

        data_url = f"data:{mime};base64,{b64}"
        messages[1]["images"] = [data_url]
        out2 = (ollama_chat(model, messages) or "").strip()
        return out2
    except Exception:
        return ""

def ingest_image(path, use_caption = True, allow_ocr = True):
    ocr = ocr_image(path) if allow_ocr else ""
    caption = caption_image_with_llava(path) if use_caption else ""
    return {
        "ocr": (ocr or "").strip(),
        "caption": (caption or "").strip(),
        "tesseract_available": bool(_HAVE_TESS),
        "tesseract_cmd": _TESS_CMD or "",
    }
