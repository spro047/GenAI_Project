# Vector Database (VDB) Implementation Plan

## 1. What is a Vector Database?
A **Vector Database** is a specialized storage system designed to manage and search through high-dimensional data, specifically **Vector Embeddings**. 

In the context of AI and LLMs:
- **Embeddings**: AI models convert words, sentences, or images into long lists of numbers (vectors) that represent their *meaning*.
- **Semantic Search**: Unlike traditional databases that look for exact word matches, a VDB finds data that is *semantically similar*. For example, searching for "canine" would successfully retrieve results about "dogs."

## 2. Why add a VDB to GraphRAG?
While GraphRAG is excellent at understanding structural relationships (e.g., "Alice is Bob's sister"), a Vector Database adds **Contextual Recall**:
1. **Fuzzy Matching**: If a user asks about "The lead character" but the graph node is labeled "Paul Atreides," the VDB can bridge that gap.
2. **Chunk Retrieval**: Sometimes the answer lies in a paragraph that didn't yield a clean graph triple. The VDB can retrieve those raw text chunks as additional "ground truth."
3. **Hybrid RAG**: Combining Graph structure with Vector search creates the most robust RAG system possible.

## 3. Technology Stack
For this project, we will use **ChromaDB**:
- **Lightweight**: Runs locally as a library.
- **Fast**: Specialized for high-speed similarity search.
- **Easy Integration**: Native Python support.

---

## 4. Step-by-Step Implementation

### Step 1: Install Dependencies
We need the vector database itself and an embedding model.
```bash
pip install chromadb sentence-transformers
```

### Step 2: Initialize the Vector Store
In `generate_kg.py`, we will initialize a persistent ChromaDB client.
```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./vdb_storage")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)
```

### Step 3: Indexing During Graph Generation
When the user clicks "Generate Graph," we will:
1. Extract triples (Graph).
2. Split the raw text into small chunks (e.g., 500 characters).
3. Store these chunks in the VDB.

### Step 4: Hybrid Retrieval in `/query`
When a user asks a question:
1. **Graph Search**: Find relevant nodes and edges (Current Logic).
2. **Vector Search**: Find top-3 most similar text chunks from the VDB.
3. **Merge**: Combine both sets of information into one "Super Context" for the LLM.

### Step 5: Final Answer Generation
The LLM will now have:
- Structural facts from the Graph.
- Detailed narrative chunks from the VDB.
Result: A 100% grounded, high-fidelity answer.

---

## 5. Next Steps
1. **Setup ChromaDB**: I will create the storage structure in the backend.
2. **Modify Extraction Pipeline**: Update the `/generate` endpoint to store vectors.
3. **Upgrade Retrieval**: Update the `/query` endpoint to perform hybrid search.
