from pdfminer.high_level import extract_text
from pathlib import Path

def load_txt(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def load_pdf(path):
    return extract_text(path)

def ingest_textlike(path):
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_txt(path)
