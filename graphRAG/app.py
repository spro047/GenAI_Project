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
from generate_kg import (
    generate_graph_from_text, 
    query_graph_rag, 
    describe_node, 
    index_text_in_vdb, 
    delete_text_from_vdb, 
    generate_graph_report,
    merge_graphs
)
import database
from fpdf import FPDF
import tempfile

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


@app.route('/delete_graph', methods=['POST'])
def delete_graph():
    """
    Deletes the chunks associated with the given text from the Vector Database.
    Expected JSON body: { "text": "original input text..." }
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data['text'].strip()
    if text:
        try:
            delete_text_from_vdb(text)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"status": "ignored"})

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

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """
    Generates a professional AI summary based on the current graph state.
    Expected JSON: { "nodes": [...], "links": [...], "communities": N }
    """
    data = request.get_json()
    if not data or 'nodes' not in data or 'links' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    nodes = data['nodes']
    links = data['links']
    communities = data.get('communities', 0)
    
    report = generate_graph_report(nodes, links, communities)
    return jsonify({"report": report})

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """
    Exports the generated report as a PDF.
    Expected JSON: { "report_md": "Markdown content", "title": "Title" }
    """
    data = request.get_json()
    if not data or 'report_md' not in data:
        return jsonify({"error": "Missing report content"}), 400
    
    report_text = data['report_md']
    title = data.get('title', 'Knowledge Graph Strategic Report')
    
    try:
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, title, align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Date
        pdf.set_font("Helvetica", 'I', 10)
        from datetime import datetime
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align='R', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        
        # Body (clean markdown roughly)
        pdf.set_font("Helvetica", size=11)
        # Simple cleanup of markdown formatting for plain PDF
        clean_text = report_text.replace('**', '').replace('###', '').replace('##', '').replace('#', '').replace('*', '-')
        
        pdf.multi_cell(0, 7, clean_text)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf.output(tmp.name)
            tmp_path = tmp.name
            
        directory = os.path.dirname(tmp_path)
        filename = os.path.basename(tmp_path)
        
        return send_from_directory(directory, filename, as_attachment=True, download_name="Knowledge_Graph_Report.pdf")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# WORKSPACE / PROJECT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/projects', methods=['GET', 'POST'])
def handle_projects():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name', 'New Project')
        p_id = database.create_project(name)
        return jsonify({"id": p_id, "name": name})
    else:
        return jsonify(database.list_projects())

@app.route('/projects/<int:project_id>', methods=['DELETE'])
def delete_project_route(project_id):
    database.delete_project(project_id)
    return jsonify({"status": "success"})

@app.route('/ingest', methods=['POST'])
def ingest_document():
    """
    Ingests a document into a project and generates its individual graph.
    """
    data = request.get_json()
    project_id = data.get('project_id')
    text = data.get('text', '').strip()
    filename = data.get('filename', 'document.txt')
    
    if not project_id or not text:
        return jsonify({"error": "Missing project_id or text"}), 400
    
    try:
        # 1. Generate Graph
        result = generate_graph_from_text(text)
        
        # 2. Save to Workspace DB
        doc_id = database.add_document(project_id, filename, text)
        database.save_graph(project_id, doc_id, result)
        
        # 3. Index in VDB for RAG
        index_text_in_vdb(text)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/merge_workspace', methods=['POST'])
def merge_workspace():
    """
    Merges all graphs within a project into one unified visualization.
    """
    data = request.get_json()
    project_id = data.get('project_id')
    
    if not project_id:
        return jsonify({"error": "Missing project_id"}), 400
        
    try:
        graphs = database.get_project_graphs(project_id)
        if not graphs:
            return jsonify({"error": "No graphs found in this project"}), 404
            
        merged = merge_graphs(graphs)
        return jsonify(merged)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    print(f"Starting Knowledge Graph Builder on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)