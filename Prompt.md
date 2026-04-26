# Recreate the Knowledge Graph Builder: Step-by-Step Guide

This document provides a comprehensive guide to recreating the **Knowledge Graph Builder**, an AI-powered tool that extracts entities and relationships from text using SerpAPI for knowledge augmentation and LLMs for triple extraction.

## 🏗️ Tech Stack
- **Backend**: Python 3.10+ (Flask)
- **Frontend**: HTML5, Vanilla CSS, D3.js (Force-directed graph)
- **APIs**: SerpAPI (Google Search), Hugging Face Inference API (Mistral/Phi-2)
- **Graph Logic**: NetworkX (for community detection)

---

## 🚀 Step 1: Environment Setup

1.  **Create a Project Directory**:
    ```bash
    mkdir knowledge-graph-builder
    cd knowledge-graph-builder
    ```

2.  **Initialize Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Create a `requirements.txt` file:
    ```text
    flask
    requests
    networkx
    python-dotenv
    google-search-results
    ```
    Install them:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🔑 Step 2: API Configuration

Create a `.env` file in the root directory:

```env
# Get from serpapi.com
SERPAPI_KEY=your_serpapi_key_here

# Get from huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=your_hf_token_here
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# LLM Fallback Settings
USE_LOCAL_LLM=false
LOCAL_LLM_URL=http://localhost:8080/v1/chat/completions
```

---

## 🧠 Step 3: Backend Logic (`generate_kg.py`)

This script handles the "brain" of the project:
1.  **Entity Extraction**: Identifying key terms in the text.
2.  **Search Augmentation**: Using SerpAPI to find more info about those terms.
3.  **LLM Triple Extraction**: Asking the LLM to output JSON triples (Subject, Predicate, Object).
4.  **Graph Construction**: Formatting nodes and edges for the frontend.

**Key Functions to Implement:**
- `search_serpapi(query)`: Uses `google-search-results` to get snippets.
- `get_augmented_text(text)`: Combines original text with search results.
- `call_hf_inference(text, model, token)`: Sends the augmented prompt to Hugging Face.
- `parse_triples_from_text(text)`: Robustly parses JSON or regex-based triples.

---

## 🌐 Step 4: Web Server (`app.py`)

Create a simple Flask server to serve the frontend and handle API requests:

```python
from flask import Flask, request, jsonify, send_from_directory
from generate_kg import generate_graph_from_text

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'knowledge_graph.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    result = generate_graph_from_text(data['text'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
```

---

## 🎨 Step 5: Frontend Interface (`knowledge_graph.html`)

The frontend uses **D3.js** to render a force-directed graph.

1.  **Layout**: A sidebar for text input and a large SVG canvas for the graph.
2.  **Graph Logic**:
    - `d3.forceSimulation`: Handles physics (attraction/repulsion).
    - `d3.zoom`: Enables panning and zooming.
    - `Tooltip`: Displays node descriptions on hover.
3.  **API Integration**: Uses `fetch` to send text to `/generate` and updates the `GRAPH_DATA` variable.

---

## 🛠️ Step 6: Running & Testing

1.  **Start the Server**:
    ```bash
    python app.py
    ```
2.  **Open in Browser**: Navigate to `http://localhost:5000`.
3.  **Test**: Paste a paragraph like *"Steve Jobs co-founded Apple in Cupertino."* and watch the graph evolve as the LLM and SerpAPI work together.

## 💡 Troubleshooting
- **Build Errors**: If `llama-cpp-python` fails to install, the code is designed to fallback to the Hugging Face API automatically.
- **Empty Graphs**: Ensure your `HUGGINGFACE_API_KEY` is valid and the model is not currently "loading" (the code handles `wait_for_model: True`).
