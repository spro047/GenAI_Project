import os
import json
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    # Create projects table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    """)
    # Create documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            filename TEXT,
            text TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    """)
    # Create graphs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            document_id INTEGER,
            graph_json TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

# Initialize database schema
init_db()

def create_project(name: str) -> int:
    """Creates a new project and returns its ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO projects (name) VALUES (?)", (name,))
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return project_id

def list_projects() -> list:
    """Returns a list of all projects."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM projects")
    rows = cursor.fetchall()
    projects = [{"id": row["id"], "name": row["name"]} for row in rows]
    conn.close()
    return projects

def delete_project(project_id: int) -> None:
    """Deletes a project and all associated documents and graphs."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM projects WHERE id = ?", (int(project_id),))
    conn.commit()
    conn.close()

def add_document(project_id: int, filename: str, text: str) -> int:
    """Adds a document to a project and returns its ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (project_id, filename, text) VALUES (?, ?, ?)",
        (int(project_id), filename, text)
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def save_graph(project_id: int, doc_id: int, graph_dict: dict) -> None:
    """Saves or updates a graph for a document."""
    conn = get_db()
    cursor = conn.cursor()
    graph_json = json.dumps(graph_dict)
    # Check if a graph already exists for this document
    cursor.execute("SELECT id FROM graphs WHERE document_id = ?", (int(doc_id),))
    row = cursor.fetchone()
    if row:
        cursor.execute(
            "UPDATE graphs SET graph_json = ? WHERE document_id = ?",
            (graph_json, int(doc_id))
        )
    else:
        cursor.execute(
            "INSERT INTO graphs (project_id, document_id, graph_json) VALUES (?, ?, ?)",
            (int(project_id), int(doc_id), graph_json)
        )
    conn.commit()
    conn.close()

def get_project_graphs(project_id: int) -> list:
    """Retrieves all graphs associated with a project."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT graph_json FROM graphs WHERE project_id = ?", (int(project_id),))
    rows = cursor.fetchall()
    graphs = []
    for row in rows:
        try:
            graphs.append(json.loads(row["graph_json"]))
        except Exception as e:
            print(f"Error parsing graph JSON: {e}")
    conn.close()
    return graphs
