# Graph-RAG (Ollama + Qdrant + Streamlit)
This project is a modular Enterprise Retrieval-Augmented Generation (RAG) prototype that supports multi-modal ingestion (text, images, audio, video), generates a connected knowledge graph, and provides hybrid search capabilities using keyword and vector-based retrieval.

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
  - **Why:** Zero API keys, runs fully local (privacy), one-line model management (ollama pull …), good Windows/macOS support, simple HTTP API. Works with both text (Mistral) and vision (LLaVA) models.
  - Alternatives:
    - OpenAI/Anthropic APIs: great quality, but require internet, billable usage, and send data off-device.
    - LM Studio / text-generation-webui: nice UIs; API/story varies, less turnkey for mixed text+vision.
    - Raw llama.cpp: very fast, but you assemble model zoo & REST glue yourself.

- **Qdrant** (Vector DB)
  - Persistent, production-ready ANN search with filters and cosine scoring.
  - Lives in Docker for easy start/stop and isolation.
  - **Why:** Production-ready ANN with filters, persistent storage, easy Docker run, great Python client, cosine/IP support, snapshotting, payload filters.
  - Alternatives:
    - Weaviate: similar feature set; Qdrant is lighter to self-host and simpler operationally for a single node.
    - FAISS: great library, but you handle persistence/sharding yourself.
    - Pinecone/Cloud DBs: managed and scalable but cloud-only and billable.

- **Streamlit** (UI)
  - Simple, reactive UI with login/register, file manager, per-user isolation, search form, and graph explorer.
  - **Why:** 10x faster to iterate than building a full frontend; live state, file uploads, forms, media, and layout with minimal code. Perfect for local tools.
  - Alternatives:
    - Gradio: fantastic for model demos; Streamlit is more flexible for multi-panel apps and stateful dashboards.
    - Flask/FastAPI + React: maximum control, but more boilerplate and slower to iterate.

- **NetworkX + PyVis** (Graph)
  - Extracted entities/relations are rendered interactively.
  - **Why:** Ultra-simple in-memory graph with rich algorithms (NetworkX) and quick interactive visualization (PyVis) without hosting a separate DB.
  - Alternatives:
    - Neo4j/Memgraph: powerful graph databases with Cypher, but add infra + deployment overhead for this use case.
    - Graphistry/Graphviz: great visualization, more ops/setup for interactive web embedding.

- **Tesseract** (OCR) + **LLaVA** (image captions)
  - Tesseract extracts text when present; LLaVA captions images for anything OCR misses.
  - **Why:** Mature, cross-platform, offline OCR; trivially integrated via pytesseract.
  - Alternatives:
    - PaddleOCR/EasyOCR: competitive accuracy and languages; larger dependency surface and GPU expectations.
    - Azure/AWS/GCP OCR: often higher accuracy, but not local and incurs costs.

- **FFmpeg / imageio-ffmpeg** (Video)
  - Extracts audio for speech-to-text and frames for OCR when needed.
  - **Why:** Rock-solid, ubiquitous media tool; imageio-ffmpeg ensures we can find a working binary reliably across systems.
  - Alternatives:
    - MoviePy/OpenCV-only: convenient but ultimately shell out to ffmpeg for many tasks; direct ffmpeg control = fewer edge cases.

- **Faster-Whisper** (Audio/Video transcription)
  - Fast, accurate local transcription.
  - **Why:** Whisper-compatible, optimized inference, good accuracy/speed locally; predictable cost (zero), no data leaves the box.
  - Alternatives:
    - OpenAI Whisper API: high quality but cloud cost & privacy concerns.
    - Vosk/DeepSpeech: lighter models; typically lower accuracy on varied domains.

- **PBKDF2 (file-based login)** (Demo auth)
  - Per-user doc isolation; each user gets their own upload dir and vector collection.
  - **Why:** Minimal dependencies, hashed passwords (PBKDF2-HMAC-SHA256), easy to audit, fits local/offline use. Each user gets isolated files and vector collections.
  - Alternatives:
    - bcrypt/argon2: stronger KDF choices; great upgrade if you plan to share the app.
    - OAuth/OIDC + DB: the right move for production multi-user deployments (sessions, TLS, ACLs).

<img width="1657" height="1069" alt="image" src="https://github.com/user-attachments/assets/86e69f13-1361-44de-94f6-02cbfc6e528f" />



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
