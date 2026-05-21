#!/usr/bin/env python3
"""
app.py

Flask server for the Knowledge Graph Builder.
Provides an API endpoint to generate knowledge graphs from text input.
"""
import os
import json
import argparse
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from fpdf import FPDF
from generate_kg import (
    generate_graph_from_text, 
    query_graph_rag, 
    describe_node, 
    index_text_in_vdb, 
    delete_text_from_vdb, 
    generate_graph_report,
    drill_down_node
)

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    """
    Serve the main knowledge graph page.
    This is the entry point of the application.
    """
    directory = os.path.abspath(os.path.dirname(__file__))
    return send_from_directory(directory, 'knowledge_graph.html')


@app.route('/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate a knowledge graph from input text.
    Expects a JSON body with a 'text' field.
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400
    
    text = data['text'].strip()
    
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        # Extract triples and build graph structure
        result = generate_graph_from_text(text)
        
        # Index the text in the Vector Database for RAG capabilities
        index_text_in_vdb(text)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete_graph', methods=['POST'])
def delete_graph():
    """
    API endpoint to delete the chunks associated with the given text from the Vector Database.
    Used when a user removes a graph from their recent history.
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
    API endpoint to query the current graph data using GraphRAG logic.
    Combines graph structural facts with semantic vector context.
    """
    data = request.get_json()
    
    if not data or 'query' not in data or 'nodes' not in data or 'links' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    q = data['query']
    nodes = data['nodes']
    links = data['links']
    history = data.get('history', [])
    
    # Process the query through the Hybrid RAG engine
    answer = query_graph_rag(q, nodes, links, history)
    return jsonify({"answer": answer})


@app.route('/describe_node', methods=['POST'])
def describe_node_route():
    """
    API endpoint to get a detailed LLM description for a specific node based on original context.
    """
    data = request.get_json()
    if not data or 'entity' not in data or 'text' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    entity = data['entity']
    text = data['text']
    desc = describe_node(entity, text)
    return jsonify({"description": desc})


@app.route('/drill_down', methods=['POST'])
def drill_down():
    """
    API endpoint to perform a deep dive discovery into a specific entity.
    Queries the VDB for niche facts and extracts a sub-graph.
    """
    data = request.get_json()
    if not data or 'entity' not in data:
        return jsonify({"error": "Missing 'entity' field"}), 400
    
    result = drill_down_node(data['entity'])
    return jsonify(result)


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """
    API endpoint to generate a professional AI strategic summary based on the current graph state.
    """
    data = request.get_json()
    if not data or 'nodes' not in data or 'links' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    nodes = data['nodes']
    links = data['links']
    communities = data.get('communities', 0)
    
    # Generate the markdown report using the LLM
    report = generate_graph_report(nodes, links, communities)
    return jsonify({"report": report})


@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """
    API endpoint to export the generated markdown report as a formatted PDF.
    Uses fpdf2 for PDF generation.
    """
    data = request.get_json()
    if not data or 'report_md' not in data:
        return jsonify({"error": "Missing report content"}), 400
    
    report_text = data['report_md']
    title = data.get('title', 'Knowledge Graph Strategic Report')
    
    try:
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Add Header
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, title, align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Add Generation Date
        pdf.set_font("Helvetica", 'I', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align='R', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        
        # Prepare Body Text (strip markdown for plain PDF rendering)
        pdf.set_font("Helvetica", size=11)
        clean_text = report_text.replace('**', '').replace('###', '').replace('##', '').replace('#', '').replace('*', '-')
        
        # Sanitize for Latin-1 encoding to prevent encoding errors with non-standard characters
        clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.multi_cell(0, 7, clean_text)
        
        # Create a temporary file to store the PDF before sending
        # We use NamedTemporaryFile and close it so fpdf can write to it on Windows
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp_path = tmp.name
            
        try:
            pdf.output(tmp_path)
            directory = os.path.dirname(tmp_path)
            filename = os.path.basename(tmp_path)
            # Send file to user for download
            return send_from_directory(directory, filename, as_attachment=True, download_name="Knowledge_Graph_Report.pdf")
        finally:
            # Note: The temp file remains on disk until system cleanup or manual deletion
            pass
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/save_graph', methods=['POST'])
def save_graph():
    """
    API endpoint to save the current state of the knowledge graph to a local JSON file.
    Used for manual overrides and persistence.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    
    try:
        # Save current graph state to disk
        with open('graph_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Start the Flask server
    port = int(os.getenv('PORT', 8000))
    print(f"Starting Knowledge Graph Builder on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)