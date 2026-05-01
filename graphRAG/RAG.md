# Implementation of GraphRAG in this Project

GraphRAG (Retrieval-Augmented Generation) combines the power of structured Knowledge Graphs with LLMs to provide more accurate and context-aware answers. Here is how it can be implemented in this project.

## 1. Current State
The project currently has a basic query mechanism in `query_graph` (`generate_kg.py`):
- It performs a simple keyword search over the extracted nodes and edges.
- It returns a template response if a direct match is found.

## 2. Proposed GraphRAG Workflow

### A. Graph-Based Retrieval
Instead of a simple keyword search, we should retrieve a "subgraph" related to the query:
1.  **Entity Linking**: Identify which entities from the Knowledge Graph are mentioned in the user's query.
2.  **Neighborhood Search**: Retrieve all direct neighbors (1-hop or 2-hop) of those entities.
3.  **Path Finding**: If multiple entities are mentioned, find the paths connecting them.

### B. Context Augmentation
Convert the retrieved graph data into a natural language context for the LLM:
- **Triple Serialization**: Convert `(Subject, Predicate, Object)` into sentences like "Subject has relationship Predicate with Object."
- **Community Context**: Use the pre-computed communities to provide broader topical context.

### C. LLM Generation
Pass the serialized context and the user query to the LLM:

```python
prompt = f"""
You are an AI assistant answering questions based on a Knowledge Graph.
CONTEXT FROM GRAPH:
{serialized_triples}

USER QUERY:
{query}

ANSWER:
"""
```

## 3. Implementation Steps

### Step 1: Enhance `query_graph`
Modify `query_graph` in `generate_kg.py` to collect context instead of returning a string directly.

```python
def get_graph_context(query, nodes, links):
    # 1. Find relevant nodes
    relevant_nodes = [n for n in nodes if n['label'].lower() in query.lower()]
    
    # 2. Get triples for those nodes
    context_triples = []
    for link in links:
        src = get_node_by_id(link['source'])
        tgt = get_node_by_id(link['target'])
        if src in relevant_nodes or tgt in relevant_nodes:
            context_triples.append(f"{src['label']} {link['label']} {tgt['label']}")
            
    return "\n".join(context_triples)
```

### Step 2: Add LLM Response Generation
Update the `/query` endpoint in `app.py` to call the LLM with the retrieved context.

```python
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    q = data['query']
    nodes = data['nodes']
    links = data['links']
    
    # Get structured context from graph
    context = get_graph_context(q, nodes, links)
    
    # Generate answer using LLM
    answer = call_llm_for_answer(q, context)
    return jsonify({"answer": answer})
```

### Step 3: Use Vector Embeddings (Optional but Recommended)
For better retrieval, use a vector database (like ChromaDB or FAISS) to store node labels and descriptions. This allows "fuzzy" matching when the user's query doesn't exactly match the entity names.

## 4. Benefits of GraphRAG
- **Explainability**: You can see exactly which nodes and edges were used to generate the answer.
- **Accuracy**: The LLM is constrained by the facts in the graph, reducing hallucinations.
- **Contextual Depth**: The graph structure reveals relationships that might be missed in flat text RAG.
