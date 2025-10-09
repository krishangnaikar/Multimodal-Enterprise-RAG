# Graph-RAG (Ollama + Qdrant + Streamlit)

A local, privacy-friendly multimodal RAG system with:
- **Ollama** for on-device LLMs (e.g., Mistral) and vision (LLaVA)
- **Qdrant** for fast, persistent vector search
- **Streamlit** UI with login, per-user docs, image captions, OCR, audio/video transcription, and graph exploration

---

## 🚀 Quick Start (Windows)

> The commands below are **Windows-only**. macOS alternatives are provided in the next section.

### 1) One-time setup (from `setup.txt`)
```powershell
# Install Ollama
# https://ollama.com/download

# Install Docker Desktop
# https://www.docker.com/products/docker-desktop/

# Install Tesseract (OCR)
# Option A: UB Mannheim installer OR Chocolatey
# choco install tesseract

# Project deps
pip install -r requirements.txt

# Install scoop (Windows package manager)
irm get.scoop.sh | iex

# Install ffmpeg using scoop
scoop install ffmpeg
```

### 2) Run the services & app
```powershell
# Start Docker Desktop
& "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe"

# Start a local Qdrant vector DB
docker run -p 6333:6333 qdrant/qdrant:latest

# Start the Streamlit app (repo root)
streamlit run app.py
```

> If your app file lives at `ui/app.py`, run:
> ```powershell
> streamlit run ui/app.py
> ```

---

## 🍎 macOS Alternatives

### Install tools
```bash
# Ollama
brew install ollama
# or download from https://ollama.com/download

# Docker Desktop (macOS)
# Download from https://www.docker.com/products/docker-desktop/
# Then start Docker.app from Applications.

# Tesseract + ffmpeg
brew install tesseract ffmpeg

# Python deps from project
pip install -r requirements.txt
```

### Run services & app
```bash
# Start Docker Desktop.app (GUI) or run once and keep it open
open -a Docker

# Qdrant
docker run -p 6333:6333 qdrant/qdrant:latest

# App
streamlit run app.py
# or, if the file is in ui/app.py:
# streamlit run ui/app.py
```

---

## 📦 Model pulls (first run)

With **Ollama** running (`ollama serve` in a separate terminal if needed):

```bash
# Text LLM (Mistral)
ollama pull mistral

# Vision model for image captions
ollama pull llava

# Embedding model
ollama pull nomic-embed-text
```

---

## 🛠 Tech Stack & Why

- **Ollama** (LLM & Vision, local)
  - `mistral`: fast, strong general-purpose reasoning for query routing, answer synthesis.
  - `llava`: captions every uploaded image for better search and graph entities.
  - `nomic-embed-text`: 768-dim embeddings for Qdrant.
  - **Why:** fully local, easy model management, no API keys, private by default.

- **Qdrant** (Vector DB)
  - Persistent, production-ready ANN search with filters and cosine scoring.
  - Lives in Docker for easy start/stop and isolation.
  - **Why:** great performance and persistence; you can restart the app without re-embedding.

- **Streamlit** (UI)
  - Simple, reactive UI with login/register, file manager, per-user isolation, search form, and graph explorer.
  - **Why:** minimal boilerplate, great for fast iteration, demos, and internal tools.

- **NetworkX + PyVis** (Graph)
  - Extracted entities/relations are rendered interactively.
  - **Why:** helps explore connections between people, projects, topics.

- **Tesseract** (OCR) + **LLaVA** (image captions)
  - Tesseract extracts text when present; LLaVA captions images for anything OCR misses.
  - **Why:** hybrid approach dramatically improves recall on slides, screenshots, photos.

- **FFmpeg / imageio-ffmpeg** (Video)
  - Extracts audio for speech-to-text and frames for OCR when needed.
  - **Why:** enables video ingestion with both transcript and on-screen text.

- **Faster-Whisper (via requirements)** (Audio/Video transcription)
  - Fast, accurate local transcription.
  - **Why:** searchable speech and graph entities from audio/video.

- **PBKDF2 (file-based login)** (Demo auth)
  - Per-user doc isolation; each user gets their own upload dir and vector collection.
  - **Why:** convenient local demo; swap for real auth/DB in production.

---

## 🧭 Features

- Per-user **login/register**; users see **only their docs**
- **File Manager**: include/exclude files (applies immediately)
- **Persistent vectors**:
  - Qdrant keeps vectors on disk
  - FAISS/NumPy fallbacks save to `state/users/<user>/`
- **Incremental rebuild**:
  - Only new/changed files get embedded
  - Excluded files get removed from search (Qdrant filter delete by `doc_id`)
- **Multimodal ingestion**:
  - Text/PDF → chunk → embed → index
  - Images → LLaVA caption + OCR → embed → index
  - Audio → transcription → embed → index
  - Video → transcript (audio) + selective frame OCR → embed → index
- **Graph Explore**: search entities & visualize neighbors
- **Hybrid retrieval**: vector + (optional) BM25 fusion, reranking, and safe answer generation

---

## 🧩 Project Structure (high-level)

```
.
├─ app.py or ui/app.py            # Streamlit app
├─ ingestion/
│  ├─ text_ingestor.py
│  ├─ image_ingestor.py           # LLaVA captions + robust Tesseract OCR
│  ├─ audio_ingestor.py           # transcription
│  └─ video_ingestor.py           # frame extraction + audio (ffmpeg)
├─ storage/
│  ├─ vector_db.py                # Qdrant/FAISS/NumPy backends, persistence
│  └─ graph_db.py                 # NetworkX + safety checks
├─ utils/
│  ├─ auth.py                     # PBKDF2 local login/register
│  ├─ persistence.py              # per-user file registry (JSON)
│  ├─ manifest.py                 # file hashes for incremental indexing
│  └─ ollama_client.py            # chat/embed helpers
└─ ui/
   └─ visualizer.py               # PyVis graph renderer
```

---

## 🔧 Troubleshooting

- **Docker pipe / engine not running (Windows)**  
  If you see errors mentioning `//./pipe/dockerDesktopLinuxEngine`:
  ```powershell
  & "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
  wsl --shutdown
  docker info
  ```
  Then re-run the Qdrant container.

- **TesseractNotFoundError**  
  Install Tesseract and/or set the path; captions still work without OCR:
  - Windows: `choco install tesseract`
  - macOS: `brew install tesseract`

- **ffmpeg not found**  
  - Windows: `scoop install ffmpeg`
  - macOS: `brew install ffmpeg`

- **Ollama returns empty captions**  
  Ensure `ollama serve` is running and models are pulled:
  ```bash
  ollama pull llava
  ollama pull mistral
  ollama pull nomic-embed-text
  ```

- **Qdrant ID validation**  
  Point IDs must be **unsigned ints** or **UUIDs** (we use numeric IDs by default).

---

## 🔐 Notes on Security

This repo includes a **file-based demo auth** (PBKDF2-hashed passwords) for local use. For production, add:
- Real authentication (OAuth/OIDC), HTTPS/TLS
- Database for users, ACLs, audit logs
- Managed vector store & object storage
- Secrets management

---

## 💡 Credits

Thanks to the open-source communities behind **Ollama**, **Qdrant**, **Streamlit**, **NetworkX/PyVis**, **Tesseract**, **FFmpeg**, and **Faster-Whisper**.
