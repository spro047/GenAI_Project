# GraphRAG — General-purpose Knowledge-Graph-from-Text Demo

This repository demonstrates a simple pipeline that converts plain text (for example, a pasted paragraph) into a small knowledge graph (entities + relationships) and exports it as JSON/HTML for visualization. It is designed as a general-purpose demo you can adapt for other domains.

## Overview

This project combines web scraping, LLM-powered entity extraction, graph analysis, and interactive visualization to build a domain-specific Retrieval-Augmented Generation (RAG) system focused on AI copyright and governance topics.

## Tech Stack

| Layer | Tools |
|-------|-------|
| LLM / RAG | Hugging Face models (via Inference API). Configure `HUGGINGFACE_MODEL` to the model you want (e.g., a Phi-family model) |
| Web scraping | SerpAPI, Trafilatura, YouTube Transcript API |
| Graph analysis | NetworkX, Graspologic (Louvain community detection) |
| Visualization | D3.js v7, vis-network 9.1.2 |
| Data | Pandas, Pydantic |


## Project Structure

```
graphrag/
├── scrape_info.ipynb              # Web scraping & text enrichment
├── graphrag_ai_copyright.ipynb    # GraphRAG pipeline, queries & visualization
├── ai_copyright_dataset.csv       # Scraped articles/videos (810 rows)
├── graph_data.json                # Extracted knowledge graph (nodes + edges)
├── ai_copyright_graph.html        # Interactive visualization (generated output)
├── graph_template.html            # HTML/D3.js template for visualization
├── .env                           # API keys (not committed)
└── lib/
    ├── bindings/utils.js          # Graph interaction utilities
    ├── vis-9.1.2/                 # vis-network library
    └── tom-select/                # Dropdown UI component
```

## Prerequisites

- Python 3.10+
- A [SerpAPI](https://serpapi.com/) key (optional, only needed for scraping notebooks)
- A Hugging Face API token for the Inference API

This demo focuses on converting pasted text into a knowledge graph. You do not need OpenAI keys anymore — the project uses the Hugging Face Inference API instead.

## Setup

1. **Clone the repo and create a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On macOS / Linux:
   # source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys / environment variables** — create a `.env` file in the project root or set these in your shell.

   Example `.env` (for demo/testing):
   ```env
   # SerpAPI key (optional)
   SERPAPI_KEY=your_serpapi_key_here

   # Hugging Face Inference API token (use the token you provided)
   HUGGINGFACE_API_KEY=your_huggingface_token_here

   # The Hugging Face model to call for triple extraction. Set to the exact model path/name you want.
   # Example: set to a Phi-family model if available on HF, e.g. HUGGINGFACE_MODEL=meta/phi-1 (replace with the real model name)
   # Recommended default: a public instruction model
   HUGGINGFACE_MODEL=microsoft/phi-2

   # Notes on using `microsoft/phi-2`:
   # 1. Visit https://huggingface.co/microsoft/phi-2 while signed in.
   # 2. Accept any model license or request access if prompted.
   # 3. Create or use a token with Inference API access and set it as HUGGINGFACE_API_KEY.
   # 4. If the hosted Inference API returns 404, the model may not be available for your token;
   #    ensure you accepted the model terms or use a model you have access to.

   # Alternative public models to try if you can't access phi-2:
   # - google/flan-t5-large
   # - google/flan-t5-base
   ```

## Usage

### Step 1 — (Optional) Scrape content (`scrape_info.ipynb`)

The original notebooks include optional scraping using SerpAPI. If you want to use scraping, provide the `SERPAPI_KEY` (above). Scraping is not required to run the core demo that converts a pasted paragraph into a knowledge graph.

### Step 2 — Generate a knowledge graph from pasted text (new script)

We added a small script `generate_kg.py` that accepts a paragraph (or text file) and:

- Calls a Hugging Face model via the Inference API to extract triples (subject, relation, object) from the text.
- Falls back to a simple heuristic entity/edge extractor if the model response can't be parsed.
- Exports `graph_data.json` suitable for the visualization template.

Files added:

- `generate_kg.py` — small command-line script to create a graph from text
- `requirements.txt` — Python dependencies for the demo

Output: `graph_data.json` (nodes + edges)

### Step 3 — Explore the visualization

Open `ai_copyright_graph.html` in a browser (or use the JSON with your visualization template). Features:

- Force-directed graph layout (D3.js)
- Filter nodes by entity type via the sidebar legend
- Search nodes by name
- Click a node to highlight its direct connections
- Hover for entity details in a tooltip
- Adjust link distance with the slider
