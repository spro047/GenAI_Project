# Knowledge Graph Builder — Workflow Report

This report documents how the project executes, step by step, from the moment a user pastes text into the browser to the rendered knowledge graph and back-and-forth Q&A. It is the executable companion to `README.md` and the visual companion to `FLOW_DIAGRAM.md`.

---

## 1. Project Overview

The application converts unstructured text into an **interactive knowledge graph** rendered with D3.js, and exposes a **Hybrid GraphRAG chatbot** that answers questions using both graph structure (extracted triples) and semantic search (ChromaDB).

**Two entry modes**:

| Mode | Triggered by | Persisted in |
|------|--------------|--------------|
| **Quick mode** | `POST /generate` | `localStorage` (browser only) |
| **Workspace mode** | `POST /ingest` | `workspace.db` (SQLite) + ChromaDB + browser |

Both modes share the same backend extraction pipeline. The only difference is persistence and the ability to merge multiple documents.

---

## 2. Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser (knowledge_graph.html, single file, ~3135 lines)            │
│  - D3.js force-directed graph                                       │
│  - Chat panel (Hybrid RAG Q&A)                                      │
│  - Workspace sidebar (projects / documents)                         │
│  - Recent graphs (localStorage)                                     │
└──────────────────────────────────────────────────────────────────────┘
                              │  fetch()  (JSON over HTTP)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Flask (app.py) — 15 routes, port 8000                              │
│  ─────────────────────────────────────                               │
│  /                       → serve knowledge_graph.html                │
│  /generate               → run quick extraction                     │
│  /delete_graph           → remove a recent from VDB                 │
│  /query                  → Hybrid RAG Q&A                           │
│  /describe_node          → LLM description of one entity            │
│  /drill_down             → VDB-driven deep dive on one entity       │
│  /generate_report        → AI strategic analysis of the graph       │
│  /export_pdf             → render markdown report as PDF            │
│  /save_graph             → manual override write to graph_data.json │
│  /projects (GET,POST)    → list / create projects                    │
│  /projects/<id> (DELETE) → delete a project (cascades)               │
│  /ingest                 → extract + save in workspace              │
│  /merge_workspace        → merge all graphs of a project            │
│  /model_info             → report the currently active model        │
└──────────────────────────────────────────────────────────────────────┘
                              │  function calls
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Core engine (generate_kg.py, ~1565 lines)                           │
│  ──────────────────────────────────────────                          │
│  generate_graph_from_text()      orchestrator (entry point)         │
│  get_augmented_text()            SerpAPI enrichment                  │
│  call_local_gguf / call_local_llm / call_hf_inference   extractors  │
│  parse_triples_from_text()       robust JSON / line parser          │
│  triples_to_graph()              node + edge construction           │
│  compute_communities()           DFS connected components            │
│  generate_graph_title()          LLM-produced one-word label         │
│  index_text_in_vdb() / delete_text_from_vdb()                       │
│  query_graph_rag()               Hybrid RAG answer                  │
│  describe_node() / drill_down_node()                                 │
│  generate_graph_report()         markdown strategic analysis         │
│  merge_graphs()                  union + dedup                       │
└──────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌────────────┐  ┌────────────┐  ┌──────────────┐
       │ ChromaDB   │  │ SQLite     │  │ External     │
       │ (VDB)      │  │ workspace  │  │ APIs         │
       │            │  │ .db        │  │              │
       │ knowledge_ │  │ projects   │  │ HuggingFace  │
       │ graph_     │  │ documents  │  │ Inference    │
       │ chunks     │  │ graphs     │  │ SerpAPI      │
       └────────────┘  └────────────┘  └──────────────┘
```

---

## 3. Workflow 1 — Generate a Knowledge Graph (Quick Mode)

**Trigger**: User pastes text into the left sidebar and clicks **Generate Graph**.

**Endpoint**: `POST /generate` (`app.py:38`)

### Step-by-step

| # | Layer | What happens | Code |
|---|-------|--------------|------|
| 1 | Browser | `fetch('/generate', { method:'POST', body: JSON.stringify({ text }) })` | `knowledge_graph.html:2506` |
| 2 | Flask | Validates the request, calls the orchestrator | `app.py:38-63` |
| 3 | Engine | **Step A — SerpAPI augmentation** (optional) | `generate_kg.py:849` |
| 3a | | `get_augmented_text(text)` extracts up to 3 capitalized entities | `generate_kg.py:212` |
| 3b | | For each entity, call `search_serpapi(entity)` and append snippets | `generate_kg.py:164` |
| 4 | Engine | **Step B — Triple extraction** (LLM, with cascading fallback) | `generate_kg.py:849-902` |
| 4a | | If `USE_LOCAL_GGUF=true` → `call_local_gguf(augmented_text)` (load GGUF file once, cached in module-level `_LOCAL_MODEL_INSTANCE`) | `generate_kg.py:114` |
| 4b | | Else if `USE_LOCAL_LLM=true` → `call_local_llm(augmented_text)` (OpenAI-compatible HTTP, e.g. Ollama / llama.cpp server) | `generate_kg.py:83` |
| 4c | | Else if `HF_API` and `HF_MODEL` set → `call_hf_inference(augmented_text, model, token)` | `generate_kg.py:249` |
| 4ci | | Inside, `call_hf_inference_with_prompt` opens `huggingface_hub.InferenceClient.chat_completion` with `max_tokens=4000` | `generate_kg.py:234` |
| 4cii | | Prompt template instructs JSON-only output of `{subject, predicate, object}` triples (no examples, explicit stop-word ban) | `generate_kg.py:251-269` |
| 4d | | If primary HF model returns "model_not_supported", try `FALLBACK_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"` | `generate_kg.py:55`, `891-898` |
| 4e | | If every LLM path fails or returns un-parseable text → `fallback_extract(text)` (regex-only, ~1-5 triples) | `generate_kg.py:500` |
| 5 | Engine | **Step C — Parse the LLM output** | `generate_kg.py:270-318` |
| 5a | | Strip markdown code fences | |
| 5b | | Try `re.search(r'\[.*\]', text, DOTALL)` and `json.loads` the array | |
| 5c | | If that fails (output was truncated, no closing `]`), extract individual `{subject, predicate, object}` objects via `re.finditer` | `generate_kg.py:294-302` |
| 5d | | Last resort: line-based ` - ` / ` \| ` / ` -> ` heuristic | `generate_kg.py:305-316` |
| 6 | Engine | **Step D — Build the graph** (`triples_to_graph`) | `generate_kg.py:320` |
| 6a | | Apply `alias_map` (resolves pronouns like "Musk" → "Elon Musk") | `generate_kg.py:313-318` |
| 6b | | For each triple, normalize the entities, skip self-loops, dedup | `generate_kg.py:424-491` |
| 6c | | Run `infer_type()` per entity: stop-word filter → person title → AI model → product → org → location → event → legislation → government → technology → PERSON (only if 2-3 capitalized words) → CONCEPT | `generate_kg.py:371-426` |
| 7 | Engine | **Step E — Community detection** (DFS over connected components) | `generate_kg.py:1108` |
| 8 | Engine | **Step F — Generate a one-word title** via LLM (`generate_graph_title`), uppercase, alphanumeric only, max 12 chars | `generate_kg.py:926` |
| 9 | Engine | **Step G — Index the text in ChromaDB** (chunked, 500-char paragraphs) for later RAG retrieval | `generate_kg.py:1232` |
| 10 | Flask | Returns `{ nodes, links, communities, extraction_method, search_augmented, title }` | `app.py:61` |
| 11 | Browser | Stores the response in `localStorage["recentGraphs"]`, renders with D3 | `knowledge_graph.html:2522-2523` |

### What the user sees in the **method** stat

The frontend fetches `/model_info` on page load (`knowledge_graph.html:2533-2548`) and shows the live active model there, so it always reflects the backend, not the cached value of an older recent graph.

---

## 4. Workflow 2 — Ask a Question (Hybrid GraphRAG)

**Trigger**: User types a question in the chat panel and presses Enter.

**Endpoint**: `POST /query` (`app.py:87`)

### Step-by-step

| # | Layer | What happens | Code |
|---|-------|--------------|------|
| 1 | Browser | Sends `{ query, nodes, links, history }` (history = last messages for follow-up) | `knowledge_graph.html` chat handler |
| 2 | Flask | Validates and calls `query_graph_rag(q, nodes, links, history)` | `app.py:87-105` |
| 3 | Engine | **Step A — Graph context**: match query words against node labels, collect 1-hop neighbors of matched nodes. Global queries (e.g. "summarize") pull top-15 nodes/edges | `generate_kg.py:1142-1211` |
| 4 | Engine | **Step B — Vector context**: `get_vector_context(query)` runs a ChromaDB similarity search, returns top-3 text chunks | `generate_kg.py:1213-1230` |
| 5 | Engine | **Step C — Build a "super context" prompt**: graph facts + vector chunks + last-5 conversation turns + strict rules (use graph for structure, text for narrative, ask 3 follow-ups at the end) | `generate_kg.py:1295-1312` |
| 6 | Engine | **Step D — LLM call** (same priority chain: GGUF → local → HF) | `generate_kg.py:1314-1335` |
| 7 | Flask | Returns `{ answer }` to the browser, which renders it as Markdown in the chat panel | `app.py:105` |

### What the LLM actually receives

```
You are an expert AI analyst and storyteller...

--- START CONTEXT ---
GRAPH CONTEXT:
- Duke Leto Atreides FOUNDED House Atreides
- Paul Atreides CHILD_OF Duke Leto Atreides
- ...

VECTOR CONTEXT:
[1] "Upon arriving on Arrakis, Paul and his family encounter a harsh environment..."
[2] "The Fremen have adapted to survive under extreme conditions..."
[3] ...
--- END CONTEXT ---

CONVERSATION HISTORY:
User: Who is Paul?
AI: Paul Atreides is the son of Duke Leto Atreides...

USER QUESTION: How does Paul become a leader?

RULES:
1. Provide a detailed, natural-sounding response.
2. Use conversation history for follow-ups.
3. Describe the 'why' and 'how' behind graph connections.
4. State politely if the answer is not in the context.
5. You MUST ask at least 3 relevant follow-up questions at the end.

DETAILED ANSWER:
```

---

## 5. Workflow 3 — Workspace / Multi-Document Mode

**Trigger**: User creates a project in the sidebar, ingests multiple documents, and clicks **Merge Workspace**.

### Step-by-step

| # | Endpoint | Effect | Persistence |
|---|----------|--------|-------------|
| 1 | `POST /projects` `{name}` | `database.create_project(name)` inserts a row, returns `{id, name}` | `workspace.db` → `projects` |
| 2 | `POST /ingest` `{project_id, text, filename}` | Runs the **full extraction pipeline** (Workflow 1, steps 3-9), then `database.add_document()` and `database.save_graph()` | `workspace.db` → `documents` and `graphs` (graph JSON in `graph_json` column) + ChromaDB |
| 3 | (repeat step 2 for more docs) | Each doc gets its own graph row | |
| 4 | `POST /merge_workspace` `{project_id}` | `database.get_project_graphs()` loads all graph JSONs, then `merge_graphs()` unions nodes (lowercase label dedup) and edges | Returned as a single graph to the browser |
| 5 | `DELETE /projects/<id>` | Cascades to documents and graphs via `ON DELETE CASCADE` | `workspace.db` |

### Merge logic (`generate_kg.py:1449`)

```
merged = {
  nodes: {<lowercase label>: {id, label, type, description, aliases[]} for unique labels},
  edges: list of (source_id, target_id, label, description), dedup on (s, t, label) tuple
}
```

---

## 6. Workflow 4 — Drill Down on a Single Node

**Trigger**: User right-clicks a node and selects **Drill down**.

**Endpoint**: `POST /drill_down` (`app.py:123`)

### Step-by-step

| # | What happens | Code |
|---|--------------|------|
| 1 | Flask calls `drill_down_node(entity_name, context_text)` | `app.py:135` |
| 2 | Engine searches ChromaDB for chunks containing the entity | `generate_kg.py:1014-1050` |
| 3 | If context is empty in both text and VDB, returns an empty sub-graph | `generate_kg.py:1051-1054` |
| 4 | Otherwise, sends a focused LLM prompt: *"Given this context, extract everything you can about `<entity>`"* | `generate_kg.py:1058-1083` |
| 5 | Falls back to `Qwen/Qwen2.5-72B-Instruct` if the primary model fails | `generate_kg.py:1084-1086` |
| 6 | Builds a sub-graph (nodes + edges) and returns it | `generate_kg.py:1014-1100` |

---

## 7. Workflow 5 — Strategic Report + PDF Export

**Trigger**: User clicks **Generate Report** in the toolbar.

### Step-by-step

| # | Endpoint | Effect | Code |
|---|----------|--------|------|
| 1 | `POST /generate_report` | Engine computes top-5 influential nodes (by degree), isolated nodes, and "bridge" nodes (high-degree cross-cluster) | `generate_kg.py:1341-1444` |
| 2 | | Sends those signals to the LLM as a prompt → returns a Markdown report | |
| 3 | Browser | Renders the Markdown in a modal | |
| 4 | `POST /export_pdf` `{report_md, title}` | Uses `fpdf2` to render the Markdown as a PDF (stripped of Markdown syntax, Latin-1 encoded) | `app.py:157-213` |
| 5 | | Streams `Knowledge_Graph_Report.pdf` as a download | |

---

## 8. Persistence Summary

| Data | Where | Lifetime | Used by |
|------|-------|----------|---------|
| Extracted graphs (Quick mode) | `localStorage["recentGraphs"]` | Browser only, max 10 entries | Recent Graphs tab |
| Graph JSON override | `graphRAG/graph_data.json` | Disk, gitignored | Manual save from frontend |
| Text chunks for RAG | `graphRAG/vdb_storage/` (ChromaDB) | Disk, gitignored | Hybrid RAG, drill-down, describe |
| Projects, documents, graphs | `graphRAG/workspace.db` (SQLite) | Disk, gitignored | Workspace mode |
| Embedded HTML (frontend) | `graphRAG/knowledge_graph.html` | Disk, **tracked in git** despite being in `.gitignore` | Served by Flask at `/` |

---

## 9. Configuration and CWD-Sensitive Env Loading

`generate_kg.py:28` runs `load_dotenv(override=True)` at import time, which reads from the **current working directory**. This makes the env file selected purely positional:

| Run from | Loads | Active model |
|----------|-------|--------------|
| `cd graphRAG && python app.py` | `graphRAG/.env` | `Qwen/Qwen2.5-72B-Instruct` |
| `python graphRAG/app.py` (from root) | `./.env` | `Qwen/Qwen2.5-72B-Instruct` |

Both `.env` files share the same working HF API key.

### LLM priority chain (highest to lowest)

1. `USE_LOCAL_GGUF=true` → load `LOCAL_GGUF_MODEL` via `llama-cpp-python` (module is **commented out** in `requirements.txt`)
2. `USE_LOCAL_LLM=true` → POST to `LOCAL_LLM_URL` (Ollama / llama.cpp server)
3. `HF_API` + `HF_MODEL` set → `huggingface_hub.InferenceClient.chat_completion` with `max_tokens=4000`
4. On `"model_not_supported"`, retry with `FALLBACK_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"`
5. Regex heuristic (`fallback_extract`) — only ~1-5 rigid patterns

The result of the run is reported back in the response field `extraction_method` and is also exposed live at `GET /model_info`.

---

## 10. Error and Fallback Behavior

| Failure point | What the user sees | Why it's safe |
|---------------|--------------------|---------------|
| HF API key expired / 401 | Logs `"Hugging Face call failed: 401 Unauthorized"`, falls through to `fallback_extract` | Response is still a valid graph (just small) |
| HF model returns non-JSON | `parse_triples_from_text` tries the truncated-array fallback, then the line-based fallback | No data is silently dropped |
| HF output exceeds `max_tokens=4000` | Truncated mid-array | The fallback parser still extracts complete `{...}` objects from the partial output |
| ChromaDB init fails | `vdb_collection = None`, the rest of the pipeline still works (RAG returns empty) | `app.py` calls are guarded with try/except |
| Vercel deployment (read-only fs) | `IS_VERCEL=1` causes VDB to use `/tmp/vdb_storage` and `ONNXMiniLM_L6_V2` embedding (no torch dep) | No code change needed |
| `GEMINI_API_KEY` / `GEMINI_MODEL` set | Nothing happens | Loaded but never used — dead config |

---

## 11. CLI Mode (Bypasses Flask)

```bash
cd graphRAG
python generate_kg.py --file "../text input/Dune.txt" --out out.json
# or
python generate_kg.py --text "Steve Jobs co-founded Apple..."
```

`main()` (`generate_kg.py:1515`) runs the same `generate_graph_from_text` orchestrator and writes a single JSON file. Useful for batch testing or scripting.

---

## 12. Cold-Start vs Warm Performance

| Operation | Cold (first request) | Warm (subsequent) |
|-----------|----------------------|-------------------|
| App boot + ChromaDB + ONNX load | ~20-30s | <1s |
| Qwen 72B extraction (single `/generate`) | 8-12s | 8-12s (model is remote) |
| Qwen 7B extraction | 2-4s | 2-4s |
| ChromaDB semantic search | <200ms | <50ms |
| D3 render of 100 nodes | ~300ms | ~300ms |

The 72B model is the bottleneck on the server side; everything else is sub-second after the first run.

---

## 13. File Map (single source of truth)

| Concern | File | Lines |
|---------|------|-------|
| Flask routes | `graphRAG/app.py` | 1-316 |
| Core engine | `graphRAG/generate_kg.py` | 1-1565 |
| SQLite workspace | `graphRAG/database.py` | 1-125 |
| Frontend | `graphRAG/knowledge_graph.html` | 1-3135 |
| Dependencies | `graphRAG/requirements.txt` | 1-10 |
| Local model (gitignored) | `graphRAG/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` | ~4 GB |
| ChromaDB storage (gitignored) | `graphRAG/vdb_storage/` | auto-created |
| Sample inputs | `text input/Dune.txt`, `text input/Harry potter.txt` | — |
| Agent instructions | `AGENTS.md` (root) | 1-58 |
| Visual flow | `graphRAG/FLOW_DIAGRAM.md` | 1-169 |
| This report | `graphRAG/REPORT.md` | 1-200+ |
