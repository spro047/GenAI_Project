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
from generate_kg import generate_graph_from_text

app = Flask(__name__, static_folder='.', static_url_path='')

# Serve the HTML file at root
@app.route('/')
def index():
    return send_from_directory('.', 'knowledge_graph.html')


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
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting Knowledge Graph Builder on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)