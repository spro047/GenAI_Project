```
┬─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM OVERVIEW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USER's BROWSER (knowledge_graph.html)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │     │
│  │  │  D3.js Graph │   │  Chat Panel  │   │  Workspace Sidebar     │  │     │ 
│  │  │  Visualizer  │   │  Q&A + Follow│   │  Projects / Docs       │  │     │
│  │  └──────┬───────┘   └──────┬───────┘   └───────────┬────────────┘  │     │
│  └─────────┼──────────────────┼───────────────────────┼───────────────┘     │
│            │                  │                       │                     │
│       POST /generate     POST /query           POST /projects               │
│            │                  │                       │                     │
├────────────┼──────────────────┼───────────────────────┼─────────────────────┤
│            ▼                  ▼                       ▼                     │
│  FLASK SERVER (app.py) ─── routes ─── calls generate_kg.py functions        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   CORE ENGINE (generate_kg.py)                      │    │
│  │                                                                     │    │
│  │  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │    │
│  │  │ Step 1   │    │  Step 2      │    │  Step 3      │               │    │
│  │  │ Augment  │───►│  Extract     │───►│  Build Graph │               │    │
│  │  │ Text     │    │  Triples     │    │  (nodes +    │               │    │
│  │  │(SerpAPI) │    │  (LLM)       │    │   edges)     │               │    │
│  │  └──────────┘    └──────────────┘    └──────┬───────┘               │    │
│  │                                              │                      │    │
│  │                                     ┌────────┴────────┐             │    │
│  │                                     │  Step 4         │             │    │
│  │                                     │  Detect Comm.   │             │    │
│  │                                     │  + Generate     │             │    │
│  │                                     │  Title          │             │    │
│  │                                     └────────┬────────┘             │    │
│  └──────────────────────────────────────────────┼──────────────────────┘    │
│                                                  │                          │
│                  ┌───────────────────────────────┼──────────────┐           │
│                  ▼                               ▼              ▼           │
│  ┌────────────────────────┐    ┌────────────────────────┐  ┌───────────┐    │
│  │  CHROMADB (Vector DB)  │    │  SQLite (Workspace DB) │  │  SerpAPI  │    │
│  │                        │    │                        │  │  Hugging  │    │
│  │  - Store text chunks   │    │  - projects table      │  │  Face API │    │
│  │  - Semantic search     │    │  - documents table     │  │  Local    │    │
│  │  - Returns relevant    │    │  - graphs table        │  │  GGUF/LLM │    │
│  │    paragraphs          │    │                        │  │           │    │
│  └────────────────────────┘    └────────────────────────┘  └───────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘



┬─────────────────────────────────────────────────────────────────────────────┐
│                  FLOW 1: GENERATE KNOWLEDGE GRAPH                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User pastes text ──► POST /generate                                        
│                                                                             │
│  1. TEXT AUGMENTATION                                                        
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  Extract capitalized entities from text                          │    │
│     │       │                                                          │    │
│     │       ▼                                                          │    │
│     │  Search SerpAPI for each entity                                  │    │
│     │       │                                                          │    │
│     │       ▼                                                          │    │
│     │  Append search results to original text                          │    │
│     └──────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│  2. TRIPLE EXTRACTION                                                       │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  Try LLMs in order until one works:                              │    │
│     │                                                                  │    │
│     │  1st: Local GGUF model (Mistral 7B on disk)                      │    │
│     │  2nd: Local LLM server (llama.cpp / Ollama)                      │    │
│     │  3rd: Hugging Face Inference API (Llama 3.3 70B)                 │    │
│     │  4th: Hugging Face fallback model                                │    │
│     │  5th: Regex heuristic (fallback_extract)                         │    │
│     │       │                                                          │    │
│     │       ▼                                                          │    │
│     │  Parse LLM output as JSON array of {subject, predicate, object}  │    │
│     └──────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│  3. GRAPH CONSTRUCTION                                                      │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  For each triple:                                                │    │
│     │    - Resolve aliases (e.g. "Musk" → "Elon Musk")                 │    │
│     │    - Normalize entity names                                      │    │
│     │    - Infer type (PERSON, ORGANIZATION, PRODUCT, LOCATION, etc.)  │    │
│     │    - Add node with {id, label, type, description}                │    │
│     │    - Add edge with {source, target, label}                       │    │
│     │    - Deduplicate edges                                           │    │
│     └──────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│  4. COMMUNITY DETECTION + TITLE                                             │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  compute_communities() via DFS connected components              │    │
│     │  generate_graph_title() via LLM prompt                           │    │
│     └──────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│  5. INDEX + RESPONSE                                                        │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  index_text_in_vdb(text) → ChromaDB chunked storage             │     │
│     │  Return {nodes, links, communities, title} to browser           │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘



┬─────────────────────────────────────────────────────────────────────────────┐
│                    FLOW 2: ASK A QUESTION (Hybrid RAG)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User types question ──► POST /query {question, nodes, links, history}      
│                                                                             │
│  1. GRAPH CONTEXT                                                          
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  Match question words against node labels                        │    │
│     │  Collect all edges involving matched nodes                       │    │
│     │  For global queries ("summarize"), include top 15 nodes + edges  │    │
│     └──────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│  2. VECTOR CONTEXT                                                          
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  ChromaDB: semantic search on question text                      │    │
│     │  Returns top 3 most relevant text chunks                         │    │
│     └──────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│  3. LLM ANSWER                                                              
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  Prompt = graph_context + vector_context + history + rules       │    │
│     │  Send to LLM (same priority chain: GGUF → Local → HF)            │    │
│     │  Returns: detailed answer + 3 follow-up questions                │    │
│     └──────────────────────────┬───────────────────────────────────────┘    │
│                                │                                            │
│  4. RESPONSE                                                                │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  Return {answer} to browser for display in Chat Panel            │    │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└──────────────────────────────────────────────────────────────────────────────



┬────────────────────────────────────────────────────────────────────────────
│                    FLOW 3: WORKSPACE (Multi-Document)                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐      ┌──────────────────┐      ┌──────────────────┐      │
│  │  CREATE PROJ │      │  INGEST DOCUMENT │      │  MERGE GRAPHS    │      │
│  │              │      │                  │      │                  │      │
│  │  POST /proj. │      │  POST /ingest    │      │  POST /merge_    │      │
│  │  {name}      │      │  {project_id,    │      │  workspace       │      │
│  │       │      │      │   text, filename}│      │  {project_id}    │      │
│  │       ▼      │      │       │          │      │       │          │      │
│  │  SQLite:     │      │       ▼          │      │       ▼          │      │
│  │  projects    │      │  generate_graph  │      │  Fetch all       │      │
│  │  table       │      │  + save in SQLite│      │  graphs from     │      │
│  │  INSERT      │      │  + index in VDB  │      │  SQLite          │      │
│  └──────────────┘      └──────────────────┘      │       │          │      │
│                                                  │       ▼          │      │
│                                                  │  Merge into      │      │
│                                                  │  unified graph   │      │
│                                                  │  (dedup nodes)   │      │
│                                                  └──────────────────┘      │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────
```