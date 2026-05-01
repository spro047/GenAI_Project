#!/usr/bin/env python3
"""
app.py

Flask server for the Knowledge Graph Builder.
Provides an API endpoint to generate knowledge graphs from text input.
"""
import os
import json
import argparse
from flask import Flask, request, jsonify, send_from_directory
from generate_kg import generate_graph_from_text, query_graph_rag, describe_node, index_text_in_vdb

app = Flask(__name__, static_folder='.', static_url_path='')

# Serve the HTML file at root
@app.route('/')
def index():
    """Serve the main knowledge graph page."""
    # Use absolute path to ensure the file is found
    directory = os.path.abspath(os.path.dirname(__file__))
    return send_from_directory(directory, 'knowledge_graph.html')


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate a knowledge graph from input text.
    
    Expected JSON body: { "text": "input paragraph here..." }
    Returns: { "nodes": [...], "links": [...], "communities": N }
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400
    
    text = data['text'].strip()
    
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        result = generate_graph_from_text(text)
        
        # Index the text in the Vector Database as well
        index_text_in_vdb(text)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    """
    Query the current graph data using GraphRAG.
    Expected JSON body: { "query": "question?", "nodes": [...], "links": [...] }
    """
    data = request.get_json()
    
    if not data or 'query' not in data or 'nodes' not in data or 'links' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    q = data['query']
    nodes = data['nodes']
    links = data['links']
    history = data.get('history', [])
    
    answer = query_graph_rag(q, nodes, links, history)
    return jsonify({"answer": answer})

@app.route('/describe_node', methods=['POST'])
def describe_node_route():
    """
    Query the LLM for a specific node's description.
    Expected JSON body: { "entity": "Name", "text": "Original context..." }
    """
    data = request.get_json()
    if not data or 'entity' not in data or 'text' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    entity = data['entity']
    text = data['text']
    desc = describe_node(entity, text)
    return jsonify({"description": desc})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    print(f"Starting Knowledge Graph Builder on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)