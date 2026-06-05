# AGENTS.md

## Run

```bash
cd graphRAG
python app.py          # starts Flask on http://localhost:8080
```

A `.venv` already exists at the repo root — activate it before running:
`.\.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (POSIX).

Port is `int(os.getenv('PORT', 8080))` at `graphRAG/app.py:334`. README's `localhost:8000` is stale — actual default is 8080.

CLI mode (no server): `python graphRAG/generate_kg.py --text "..."` or `--file in.txt --out out.json`.

## Configuration

- **Two `.env` files** with different keys:
  - `./.env` (root): `HUGGINGFACE_API_KEY`, `HUGGINGFACE_MODEL`, `LOCAL_LLM_URL`, `USE_LOCAL_LLM`, `USE_LOCAL_GGUF`, `SERPAPI_KEY`
  - `graphRAG/.env`: only `HUGGINGFACE_API_KEY`, `HUGGINGFACE_MODEL`, `SERPAPI_KEY` (no local-LLM flags)
- `graphRAG/generate_kg.py:28`: `load_dotenv(override=True)` reads `.env` from CWD. Run from `graphRAG/` to use `graphRAG/.env`; run from root to use `./.env`.
- Both `.env` files currently set `HUGGINGFACE_MODEL=Qwen/Qwen2.5-72B-Instruct`. Both share one working HF token; SerpAPI keys differ.
- LLM selection priority (first non-empty wins): `USE_LOCAL_GGUF=true` → `USE_LOCAL_LLM=true` → `HUGGINGFACE_API_KEY` → regex `fallback_extract()`.
- `GEMINI_API_KEY` / `GEMINI_MODEL` are read at `generate_kg.py:41-42` but never used. Dead config.

## Architecture

| File | Role |
|------|------|
| `graphRAG/app.py` (307 lines) | Flask server, 14 routes (incl. `/projects`, `/ingest`, `/merge_workspace`, `/model_info`) |
| `graphRAG/generate_kg.py` (~1396 lines) | Core engine: extraction, ChromaDB, graph build, RAG, PDF report, community detection |
| `graphRAG/database.py` (115 lines) | SQLite `workspace.db` (projects, documents, graphs) — `init_db()` runs at import, line 48 |
| `graphRAG/knowledge_graph.html` (~2786 lines) | Single-page frontend: D3.js graph, chatbot, workspace sidebar, all CSS/JS embedded |
| `graphRAG/REPORT.md`, `graphRAG/FLOW_DIAGRAM.md` | Step-by-step workflow docs (line counts in them are also stale) |

- **No tests, no CI, no linter/formatter/typechecker.**
- **No build step.** Frontend = the one HTML file, served by Flask.
- ChromaDB persists to `graphRAG/vdb_storage/` locally; auto-routes to `/tmp/vdb_storage` when `VERCEL=1` (`generate_kg.py:61-62`).
- SQLite `graphRAG/workspace.db` and `graphRAG/graph_data.json` are auto-generated and gitignored.
- `knowledge_graph.html` IS tracked in git (despite the `.gitignore` entry — ignore is a no-op for tracked files). Do not `git rm` without confirming.
- `numpy<2.0.0` pinned in `graphRAG/requirements.txt` — upgrading breaks ChromaDB.
- `llama-cpp-python` is **commented out** in `requirements.txt`; uncomment for `USE_LOCAL_GGUF=true` mode (also requires C++ build tools).
- Logging is hardcoded to `DEBUG` at `generate_kg.py:24` — not env-configurable.

## API surface (Flask)

Routes are flat — no blueprints, no CORS config (frontend is same-origin). Frontend calls all paths as relative URLs.

Quick mode (browser-only, no DB): `POST /generate` (text in → graph out), `POST /delete_graph`, `POST /save_graph`.

RAG / analysis: `POST /query` (graph + vector hybrid), `POST /describe_node`, `POST /drill_down`, `POST /generate_report`, `POST /export_pdf`.

Workspace (persisted): `GET|POST /projects`, `DELETE /projects/<id>`, `POST /ingest`, `POST /merge_workspace`.

Diagnostic: `GET /model_info` — returns the active model + flags.

## Quirks

- **Two HF calling paths exist** in `generate_kg.py`:
  1. `requests.post` via `huggingface_hub.InferenceClient.chat_completion` — used for triple extraction (`call_hf_inference` → `call_hf_inference_with_prompt`).
  2. Same `InferenceClient` pattern used inline for `query_graph_rag` (`generate_kg.py:1326`) and `generate_graph_report` (line 1435).
  Both can fail independently; check both in logs.
- **Fallback model swap**: if primary `HUGGINGFACE_MODEL` returns `model_not_supported`, code auto-retries with `FALLBACK_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"` (`generate_kg.py:55, 894`).
- **VDB chunk IDs are derived from text** (`generate_kg.py:1249`): `chunk_<first10_chars_alphanumeric>_<index>`. Two texts starting with the same first 10 alphanumeric chars will collide on chunk IDs.
- **VDB chunking is naive**: split on `\n\n`, then 800-char sliding with 200-char overlap if only one paragraph. Long single-paragraph inputs lose structure.
- **SerpAPI augments before LLM call** for entity names extracted by a capitalized-noun regex (`extract_key_entities`); only the top 3 entities are searched.
- Sample inputs: `text input/Dune.txt`, `text input/Harry potter.txt`.
- GGUF model at `graphRAG/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` (~4GB, gitignored). Set `LOCAL_GGUF_MODEL` env var to its absolute path; `USE_LOCAL_GGUF=true` to enable.
- ChromaDB collection name = `knowledge_graph_chunks`, distance = `cosine`.
- Workspace SQLite DB is initialized on import (`database.py:48`), not on first request.

## Graph quality troubleshooting

Graph density depends on which extraction method fires (worst → best):

1. **Regex heuristic** (`fallback_extract`, `generate_kg.py:500`) — strict, role-based; ~1–10 triples; `extraction_method="Heuristic Fallback"`.
2. **HuggingFace API** (default) — quality varies with the configured `HUGGINGFACE_MODEL`.
3. **Local LLM** (OpenAI-compatible) — good if endpoint is reachable.
4. **Local GGUF** — best local quality; needs `llama-cpp-python`.

If graphs come back sparse, the HF call is probably failing. Check console for `"Hugging Face call failed"` or `"Model ... not supported"`. The `extraction_method` and `search_augmented` fields in the `/generate` response tell you which path fired and whether SerpAPI added context.

## Vercel deployment

- `vercel.json` rewrites all `/*` to `graphRAG/app.py` via `@vercel/python` (runtime `python3.12`, `maxLambdaSize: 300mb`).
- **Env vars must be set in Vercel Dashboard**, not `.env`: `HUGGINGFACE_API_KEY`, `HUGGINGFACE_MODEL`, `SERPAPI_KEY`.
- ChromaDB auto-uses `/tmp/vdb_storage` on Vercel; storage is ephemeral (lost between cold starts).
- Embedding function: `ONNXMiniLM_L6_V2` on Vercel (no torch), `DefaultEmbeddingFunction` (SentenceTransformers) locally.
- `workspace.db` does **not** persist on Vercel — `/projects`, `/ingest`, `/merge_workspace` won't survive across requests.
- **Cold start is slow** (~20–30s) due to ChromaDB + ONNX + numpy init.
