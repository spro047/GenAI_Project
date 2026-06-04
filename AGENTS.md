# AGENTS.md

## Run

```bash
cd graphRAG
python app.py          # starts Flask on http://localhost:8000
```

## Configuration

- Two `.env` files exist: `./.env` and `graphRAG/.env`. They contain **different** keys and model settings. The root `.env` has `USE_LOCAL_LLM`, `LOCAL_LLM_URL`, `USE_LOCAL_GGUF` keys that `graphRAG/.env` lacks.
- `generate_kg.py:28`: `load_dotenv(override=True)` loads `.env` from CWD. Running from `graphRAG/` uses `graphRAG/.env`; running from root uses `./.env`.
- Root `.env` and `graphRAG/.env` both use `Qwen/Qwen2.5-72B-Instruct` for detailed extraction. Both share the same working key.
- LLM selection priority: GGUF direct → local OpenAI-compatible → HuggingFace Inference API (Qwen 72B) → regex heuristic fallback.
- `GEMINI_API_KEY` and `GEMINI_MODEL` are loaded from env but **never used** — dead config.

## Architecture

| File | Role |
|------|------|
| `graphRAG/app.py` | Flask server, defines all API routes |
| `graphRAG/generate_kg.py` (~1562 lines) | Core engine: LLM extraction, ChromaDB, graph building, RAG querying |
| `graphRAG/database.py` | SQLite workspace DB (projects, documents, graphs tables) |
| `graphRAG/knowledge_graph.html` (~3135 lines) | Single-page frontend: D3.js graph, chatbot, all CSS/JS embedded |

- **No tests / CI / linter / formatter / type checker.**
- **No build step.** Frontend = single HTML file served by Flask.
- ChromaDB persists to `graphRAG/vdb_storage/` (auto-created, gitignored).
- SQLite DB `graphRAG/workspace.db` and graph JSON `graphRAG/graph_data.json` auto-generated and gitignored.
- `numpy<2.0.0` pinned in `requirements.txt` — upgrading breaks ChromaDB.
- `llama-cpp-python` is **commented out** in `requirements.txt`; must uncomment for GGUF mode.
- Logging is hardcoded to `DEBUG` at module level (`generate_kg.py:24`), not env-configurable.

## Quirks

- Flask runs on **port 8000** (not 5000), configurable via `PORT` env var.
- Sample inputs: `text input/Dune.txt`, `text input/Harry potter.txt`.
- GGUF model at `graphRAG/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` (~4GB, gitignored).
- ChromaDB collection name = `knowledge_graph_chunks` with `hnsw:space: cosine`.
- Two HF calling paths: `requests.post` (triple extraction via `call_hf_inference`) vs `huggingface_hub.InferenceClient.chat_completion` (RAG query via `query_graph_rag`).
- CLI mode: `python generate_kg.py --text "..."` or `--file in.txt --out out.json`.
- No migrations — `database.py:init_db()` auto-creates schema on startup.

## Graph Quality Troubleshooting

Graph detail depends on which extraction method fires:
1. GGUF (if `USE_LOCAL_GGUF=true`) — best quality, local
2. Local LLM (if `USE_LOCAL_LLM=true`) — good quality, local
3. **HuggingFace API** (default) — quality depends on model
4. Regex heuristic fallback — minimal, ~1-5 triples

If graphs become sparse, the HF API call is likely failing (expired key, rate limit, model access revoked). Check console for `"Hugging Face call failed"` or `"falling back to heuristic"`. The `extraction_method` field in the response tells you which method fired.

## Vercel Deployment

- `vercel.json` routes all traffic (`/*`) to `graphRAG/app.py` via `@vercel/python`.
- **Set env vars in Vercel Dashboard** (not `.env` — those are local only): `HUGGINGFACE_API_KEY`, `HUGGINGFACE_MODEL`, `SERPAPI_KEY`.
- `knowledge_graph.html` is gitignored but **tracked** in git. Do not delete without confirming intent.
- ChromaDB auto-uses `/tmp/vdb_storage` on Vercel (detected via `VERCEL=1` env var). Storage is ephemeral.
- `ONNXMiniLM_L6_V2` embedding on Vercel (no torch dep); `DefaultEmbeddingFunction` (SentenceTransformers) locally.
- Workspace SQLite DB (`workspace.db`) does **not** persist on Vercel. Project features won't work across requests.
- **Cold start is slow** (~20-30s) due to ChromaDB/ONNX/numpy deps.
