# Knowledge Graph Builder & Hybrid GraphRAG

This project is a powerful, local-first web application that converts unstructured text into interactive Knowledge Graphs. It features a built-in conversational AI (GraphRAG) that uses both graph structures and semantic vector search to answer questions about your data.

## 🚀 Features

*   **Automated Knowledge Extraction**: Paste any text, and the backend uses an LLM (Hugging Face or Local) to extract entities and relationships (triples).
*   **Interactive Visualization**: A premium, dark-themed UI powered by **D3.js** allows you to explore the graph. Features include zooming, panning, node dragging, clustering, and filtering by entity type.
*   **Hybrid GraphRAG Chatbot**: A conversational assistant that understands your data.
    *   **Graph Context**: Uses extracted relationships to answer structural questions (e.g., "Who is connected to X?").
    *   **Vector Context**: Integrates **ChromaDB** to retrieve narrative context for fuzzy matching and deeper explanations.
    *   **Premium UI**: Markdown rendering, chat history for follow-up questions, and "Copy/Edit" action buttons.
*   **Recent Graphs Tab**: Automatically saves your generated graphs to `localStorage` so you can instantly switch between recent analyses without re-processing.
*   **Local-First Design**: Supports connecting to local LLMs (like Ollama or llama.cpp) and stores all vector embeddings locally, ensuring your data never leaves your machine unless you use a cloud LLM.

---

## 🛠️ Tech Stack

*   **Backend**: Python, Flask
*   **Graph Processing**: NetworkX, Graspologic (Louvain community detection)
*   **Vector Database**: ChromaDB, Sentence-Transformers
*   **Frontend**: HTML5, Vanilla CSS, Vanilla JavaScript, D3.js (v7)
*   **AI Integration**: Hugging Face Inference API (default), OpenAI-compatible local endpoints, or direct GGUF loading.

---

## 📁 Project Structure

```
graphRAG/
├── app.py                      # Flask server & API endpoints (/generate, /query)
├── generate_kg.py              # LLM extraction logic, GraphRAG engine, ChromaDB setup
├── knowledge_graph.html        # Main frontend UI (D3 visualization, Chatbot, Sidebar)
├── requirements.txt            # Python dependencies
├── .env                        # API keys and configuration
├── .gitignore                  # Git ignore rules (ignores VDB storage)
├── RAG.md                      # Documentation on RAG implementation
├── VDB.md                      # Documentation on Vector DB implementation
└── vdb_storage/                # Local ChromaDB persistent storage (Auto-generated)
```

---

## ⚙️ Setup & Installation

1. **Clone and setup a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables (`.env`):**
   Create a `.env` file in the root directory. 
   ```env
   # Default: Use Hugging Face
   HUGGINGFACE_API_KEY=your_token_here
   HUGGINGFACE_MODEL=meta-llama/Llama-3.3-70B-Instruct

   # Optional: Use Local LLM (e.g., Ollama)
   USE_LOCAL_LLM=false
   LOCAL_LLM_URL=http://localhost:11434/v1/chat/completions
   ```

---

## 🏃‍♂️ Usage

1. **Start the Server**:
   ```bash
   python app.py
   ```
2. **Access the UI**:
   Open your browser and navigate to `http://localhost:8000`.

3. **Generate a Graph**:
   Paste a story or article into the left sidebar and click "Generate Graph". The system will extract the nodes, store the text chunks in ChromaDB, and render the visualization.

4. **Chat with your Data**:
   Click the chat icon in the bottom right corner to open the GraphRAG assistant. Ask questions like:
   * "Summarize the entire graph."
   * "How is Character A related to Character B?"
   * "What happened in the original text regarding [Entity]?"

---

## 🧠 How Hybrid GraphRAG Works

When you ask a question in the chatbot, the system executes a two-pronged retrieval strategy:
1.  **Keyword/Node Matching**: Identifies entities in your query and pulls their direct connections from the visual graph.
2.  **Semantic Search (ChromaDB)**: Converts your question into an embedding and retrieves the top 3 most relevant raw text chunks from your original input.

The LLM is then provided with a "Super Context" containing both the structural facts and the narrative paragraphs, allowing it to generate highly accurate, detailed, and formatted responses.
