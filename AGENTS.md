# AGENTS.md

## Run

```bash
cd graphRAG
python app.py          # starts Flask on http://localhost:8000
```

## Configuration

- Two `.env` files exist: `./.env` and `graphRAG/.env`. They contain **different** API keys and model settings. The root `.env` has `USE_LOCAL_LLM`, `LOCAL_LLM_URL`, `USE_LOCAL_GGUF` keys that `graphRAG/.env` lacks.
- `generate_kg.py:21` uses `load_dotenv(override=True)` — it loads `.env` from CWD.
- LLM selection priority (highest to lowest): GGUF direct → local OpenAI-compatible → HuggingFace Inference API → regex heuristic fallback.

## Architecture

| File | Role |
|------|------|
| `graphRAG/app.py` | Flask server, defines all API routes |
| `graphRAG/generate_kg.py` (~1553 lines) | Core engine: LLM extraction, ChromaDB, graph building, RAG querying |
| `graphRAG/database.py` | SQLite workspace DB (projects, documents, graphs tables) |
| `graphRAG/knowledge_graph.html` (~3135 lines) | Single-page frontend: D3.js graph, chatbot, all CSS/JS embedded |

- **No test files, no test framework, no CI, no linter, no formatter, no type checker.**
- **No build step.** The frontend is a single HTML file served directly by Flask.
- ChromaDB persists vector storage to `graphRAG/vdb_storage/` (auto-created, gitignored).
- SQLite DB `graphRAG/workspace.db` and graph JSON `graphRAG/graph_data.json` are auto-generated and gitignored.
- `knowledge_graph.html` is in `.gitignore` but is **tracked in git** (committed before gitignore was added). Do not remove from git without confirming intent.

## Quirks

- Flask runs on **port 8000** (not 5000), configurable via `PORT` env var.
- `generate_kg.py` is the file to modify for LLM extraction logic, ChromaDB operations, or RAG strategy.
- `app.py` is the file to modify for API routes or server behavior.
- Sample test inputs are in `text input/Dune.txt` and `text input/Harry potter.txt`.
- GGUF model at `graphRAG/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` is ~4GB and gitignored.
- No migrations — `database.py:init_db()` auto-creates schema on startup.

## Vercel Deployment

- `vercel.json` routes all traffic (`/*`) to `graphRAG/app.py` via `@vercel/python`.
- **Set env vars in Vercel Dashboard** (not `.env` — those are local only): `HUGGINGFACE_API_KEY`, `HUGGINGFACE_MODEL`, `SERPAPI_KEY`. The `.env` files are gitignored and won't deploy.
- ChromaDB auto-uses `/tmp/vdb_storage` on Vercel (detected via `VERCEL=1` env var). Storage is ephemeral — data resets per deployment.
- `ONNXMiniLM_L6_V2` embedding used on Vercel (lighter, no torch dep); `DefaultEmbeddingFunction` (SentenceTransformers) used locally.
- `knowledge_graph.html` is gitignored but **tracked** — it will deploy with git push. Do not delete it.
- Workspace SQLite DB (`workspace.db`) does **not** persist on Vercel (ephemeral filesystem). Workspace/project features won't work across requests.
- **Cold start is slow** (~20-30s) due to ChromaDB/ONNX/numpy deps. Subsequent requests are fast.
