from __future__ import annotations
import os
from pathlib import Path
import streamlit as st

from utils.chunking import chunk_text, enrich_metadata
from ingestion.text_ingestor import ingest_textlike
from ingestion.image_ingestor import ingest_image
from ingestion.audio_ingestor import transcribe_audio
from ingestion.video_ingestor import extract_frames, extract_audio
from storage.vector_db import VectorStore
from storage.graph_db import GraphStore
from nlp.entity_extractor import extract_triples
from retrieval.query_router import route_query
from retrieval.hybrid_search import keyword_search, fuse_results
from retrieval.reranker import topical_rerank
from generation.answer_generator import generate_answer
from generation.postprocessor import sanitize_answer
from ui.visualizer import render_pyvis_graph

from utils.auth import register_user, authenticate_user
from utils.persistence import load_file_index, save_file_index
from utils.manifest import load_manifest, save_manifest, file_hash

st.set_page_config(page_title="EnerpriseGPT", layout="wide")
st.title("EnerpriseGPT")

st.session_state.setdefault("user", None)
st.session_state.setdefault("file_index", {})
st.session_state.setdefault("manifest", {})
st.session_state.setdefault("boot_rebuilt", False)

st.session_state.setdefault("answer", "")
st.session_state.setdefault("contexts", [])
st.session_state.setdefault("last_query", "")
st.session_state.setdefault("route", {})
st.session_state.setdefault("graph_nodes", [])
st.session_state.setdefault("img_captions", [])
st.session_state.setdefault("stats", {"files": 0, "chunks": 0, "skipped": 0})

def _uploads_root_for(user):
    d = Path("uploads") / user
    d.mkdir(parents=True, exist_ok=True)
    return d

def _bump_stat(key, inc = 1):
    s = st.session_state["stats"]
    s[key] = s.get(key, 0) + inc

def _register_files_for_user(user, files):
    if not files:
        return
    idx = st.session_state["file_index"]
    root = _uploads_root_for(user)
    for uf in files:
        dst = root / uf.name
        os.makedirs(dst.parent, exist_ok=True)
        dst.write_bytes(uf.read())
        abs_path = str(dst.resolve())
        suffix = dst.suffix.lower()
        modality = (
            "text"  if suffix in [".pdf", ".txt"] else
            "image" if suffix in [".jpg", ".jpeg", ".png"] else
            "audio" if suffix in [".mp3", ".wav", ".m4a"] else
            "video" if suffix in [".mp4", ".mov", ".mkv"] else
            "other"
        )
        idx[abs_path] = {"included": True, "doc_id": uf.name, "modality": modality}
    save_file_index(user, idx)

def _ingest_text_block(doc_id, modality, p, text):
    if not text:
        return 0
    chunks = chunk_text(text)
    payloads = [enrich_metadata(doc_id, modality, str(p), {"chunk_id": i}) for i, _ in enumerate(chunks)]
    st.session_state.vec.index_texts(chunks, payloads)
    ents, rels = extract_triples(text)
    if ents or rels:
        st.session_state.graph.add_entities_relations(ents, rels)
    return len(chunks)

def rebuild_from_selection(user, extract_every = 60, transcribe_video_audio = True, use_caption = True):
    idx = st.session_state["file_index"]
    manifest = st.session_state["manifest"]
    vec = st.session_state.vec

    st.session_state["stats"] = {"files": 0, "chunks": 0, "skipped": 0}
    st.session_state["img_captions"] = []
    st.session_state["graph"] = GraphStore()

    if vec.backend == "qdrant":
        remove_paths = []
        for abs_path, meta in manifest.items():
            if abs_path not in idx or not idx.get(abs_path, {}).get("included", False):
                doc_id = meta.get("doc_id", Path(abs_path).name)
                try:
                    vec.delete_by_doc(doc_id)
                except Exception:
                    pass
                remove_paths.append(abs_path)
        for p in remove_paths:
            manifest.pop(p, None)

    for abs_path, info in idx.items():
        if not info.get("included"):
            continue
        p = Path(abs_path)
        if not p.exists():
            _bump_stat("skipped")
            continue

        doc_id = info.get("doc_id", p.name)
        modality = info.get("modality", "text")
        suffix = p.suffix.lower()

        h = file_hash(abs_path)
        cached = manifest.get(abs_path, {})
        if cached.get("hash") == h:
            continue  # nothing to do

        added_chunks = 0
        try:
            if suffix in [".pdf", ".txt"]:
                text = ingest_textlike(str(p))
                added_chunks += _ingest_text_block(doc_id, "text", p, text)
                if added_chunks > 0:
                    _bump_stat("files")

            elif suffix in [".jpg", ".jpeg", ".png"]:
                pack = ingest_image(str(p), use_caption=use_caption, allow_ocr=True)
                caption = (pack.get("caption") or "").strip()
                ocr = (pack.get("ocr") or "").strip()

                if caption:
                    cap_chunks = _ingest_text_block(doc_id, "image", p, f"[CAPTION] {caption}")
                    added_chunks += cap_chunks
                    st.session_state.img_captions.append({"path": str(p), "caption": caption})
                if ocr:
                    added_chunks += _ingest_text_block(doc_id, "image", p, f"[OCR] {ocr}")
                if added_chunks > 0:
                    _bump_stat("files")

            elif suffix in [".mp3", ".wav", ".m4a"]:
                text = transcribe_audio(str(p))
                added_chunks += _ingest_text_block(doc_id, "audio", p, text)
                if added_chunks > 0:
                    _bump_stat("files")

            elif suffix in [".mp4", ".mov", ".mkv"]:
                frame_dir = p.parent / f"{p.stem}_frames"
                frames = extract_frames(str(p), str(frame_dir), every_n=int(extract_every))
                from ingestion.image_ingestor import ocr_image
                parts = []
                for f in frames[:20]:
                    t = ocr_image(f)
                    if t.strip():
                        parts.append(t)
                if transcribe_video_audio:
                    try:
                        audio_out = p.parent / f"{p.stem}.mp3"
                        audio_path = extract_audio(str(p), str(audio_out), codec="mp3", overwrite=True)
                        tr = transcribe_audio(audio_path)
                        if tr.strip():
                            parts.append("\n[TRANSCRIPT]\n" + tr.strip())
                    except Exception:
                        pass
                text = "\n".join(parts)
                added_chunks += _ingest_text_block(doc_id, "video", p, text)
                if added_chunks > 0:
                    _bump_stat("files")
            else:
                _bump_stat("skipped")
        except Exception:
            _bump_stat("skipped")

        if added_chunks > 0:
            _bump_stat("chunks", added_chunks)
            manifest[abs_path] = {"hash": h, "modality": modality, "doc_id": doc_id}
        else:
            if vec.backend == "qdrant":
                try:
                    vec.delete_by_doc(doc_id)
                except Exception:
                    pass
            manifest.pop(abs_path, None)

    save_manifest(user, manifest)
    st.session_state["manifest"] = manifest
    if vec.backend in ("faiss", "numpy"):
        vec.save_local(user)

def login_register_view():
    st.subheader("Sign in or create an account")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Login"):
            ok, msg = authenticate_user(u, p)
            if ok:
                user = u.strip().lower()
                st.session_state["user"] = user
                st.session_state["file_index"] = load_file_index(user)
                st.session_state["manifest"] = load_manifest(user)
                st.session_state["vec"] = VectorStore(use_qdrant=True, collection=f"mmrag_{user}", dim=768)
                st.session_state["vec"].load_local(user)
                st.session_state["graph"] = GraphStore()
                st.success("Logged in.")
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        u2 = st.text_input("New username", key="reg_u")
        p2 = st.text_input("New password", type="password", key="reg_p")
        if st.button("Register"):
            ok, msg = register_user(u2, p2)
            if ok:
                st.success(msg + " Please login.")
            else:
                st.error(msg)

if not st.session_state["user"]:
    login_register_view()
    st.stop()

user = st.session_state["user"]

if "vec" not in st.session_state:
    st.session_state["vec"] = VectorStore(use_qdrant=True, collection=f"mmrag_{user}", dim=768)
    st.session_state["vec"].load_local(user)
if "graph" not in st.session_state:
    st.session_state["graph"] = GraphStore()
if "corpus" not in st.session_state:
    st.session_state["corpus"] = []
if "meta" not in st.session_state:
    st.session_state["meta"] = []

if not st.session_state["boot_rebuilt"]:
    rebuild_from_selection(user)
    st.session_state["boot_rebuilt"] = True

with st.sidebar:
    st.write(f"**Signed in:** `{user}`")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.header("File Manager")
    idx = st.session_state["file_index"]
    if idx:
        st.caption("Toggle which files are included (applies immediately):")
        paths_sorted = sorted(idx.keys(), key=lambda p: Path(p).name.lower())
        changed = False
        for pth in paths_sorted:
            info = idx[pth]
            label = f"{Path(pth).name}  ({info.get('modality','?')})"
            key = f"inc::{pth}"
            current = bool(info.get("included", True))
            new_val = st.checkbox(label, value=current, key=key)
            if new_val != current:
                idx[pth]["included"] = bool(new_val)
                changed = True
        if changed:
            save_file_index(user, idx)
            rebuild_from_selection(user)
            st.toast("Selection updated & index rebuilt")
    else:
        st.caption("No files yet. Upload below.")

    st.divider()
    st.header("Ingest New Files")
    use_caption = st.checkbox("Use LLAVA captions for images", value=True, key="use_caption")
    extract_every = st.number_input("Video: extract every N frames", min_value=30, max_value=300, value=60, step=10, key="extract_every")
    transcribe_video_audio = st.checkbox("Transcribe video audio", value=True, key="tx_video")

    uploads = st.file_uploader(
        "Upload files",
        type=["pdf","txt","jpg","jpeg","png","mp3","wav","m4a","mp4","mov","mkv"],
        accept_multiple_files=True
    )
    if uploads:
        _register_files_for_user(user, uploads)
        rebuild_from_selection(user, extract_every=int(extract_every),
                               transcribe_video_audio=bool(transcribe_video_audio),
                               use_caption=bool(use_caption))
        s = st.session_state["stats"]
        st.success(f"Files added & indexed  | Files: {s['files']} | Chunks: {s['chunks']} | Skipped: {s['skipped']}")

active = [Path(p).name for p, info in st.session_state["file_index"].items() if info.get("included")]
if active:
    st.caption("**Active documents:** " + ", ".join(active))
else:
    st.warning("No active documents selected.")

st.header("Ask a Question")
with st.form("search_form"):
    q = st.text_input("Enter a query", key="q_input")
    topic_hint = st.text_input("Optional topic hint for reranker (keywords)", key="hint_input")
    submitted = st.form_submit_button("Search")
    if submitted and q:
        route = route_query(q)
        q2 = route.get("rewrite", q)

        vec_res = st.session_state.vec.search(q2, k=8)
        kw_res = keyword_search(st.session_state.get("corpus", []), q2, k=8) if st.session_state.get("corpus") else []

        fused = fuse_results(vec_res, kw_res)
        reranked = topical_rerank(fused, topic_hint=topic_hint)

        if not reranked:
            st.session_state.answer = "I couldn’t find matching context. Try different keywords or include more documents."
            st.session_state.contexts = []
        else:
            ans = generate_answer(q, reranked[:6])
            st.session_state.answer = sanitize_answer(ans)
            st.session_state.contexts = reranked[:6]

        st.session_state.last_query = q
        st.session_state.route = route

if st.session_state.answer:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Answer")
        st.write(st.session_state.answer)

        st.subheader("Top Contexts")
        for r in st.session_state.contexts:
            with st.expander(f"{r.get('modality','?')} | {r.get('source','?')} | score={r.get('score',0):.3f}"):
                st.write((r.get("text") or "")[:1200])

    with col2:
        st.subheader("Graph Explore")
        with st.form("graph_form"):
            entity_q = st.text_input("Find entity by name", key="entity_q")
            depth = st.slider("Neighbor depth", 1, 3, 1, key="depth_slider")
            gsubmit = st.form_submit_button("Search Entity")

        if gsubmit and entity_q:
            hits = st.session_state.graph.search_entities_by_name(entity_q)
            if hits:
                st.write("Matches:", hits[:5])
                nid = hits[0]["id"]
                G = st.session_state.graph.G
                layer, seen = {nid}, {nid}
                for _ in range(depth):
                    nxt = set()
                    for u in layer:
                        nxt.update(G.successors(u))
                        nxt.update(G.predecessors(u))
                    layer = nxt - seen
                    seen.update(layer)
                st.session_state.graph_nodes = list(seen)
            else:
                st.info("No matching entities.")
                st.session_state.graph_nodes = []

        if st.session_state.graph_nodes:
            render_pyvis_graph(st.session_state.graph.G, st.session_state.graph_nodes)
