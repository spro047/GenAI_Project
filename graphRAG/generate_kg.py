#!/usr/bin/env python3
"""
generate_kg.py

Simple demo script that: given a paragraph of text, calls a Hugging Face model
via the Inference API to extract triples (subject, relation, object). Falls back
to a simple heuristic if the model output can't be parsed as JSON.

Outputs `graph_data.json` with `nodes` and `edges` suitable for the existing
visualization template.
"""
import os
import sys
import argparse
import json
import requests
import re
from collections import defaultdict


HF_API = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL")
# Fallback public model to try if the configured model path is not found
FALLBACK_HF_MODEL = "google/flan-t5-large"


def call_hf_inference(text: str, model: str, token: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    prompt = (
        "Extract triples from the following text as a JSON list of objects with keys "
        '"subject", "predicate", "object".\n\nText:\n' + text
    )
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    # Many HF models return a list of generated items or a dict with 'generated_text'
    try:
        data = resp.json()
    except Exception:
        return resp.text
    # If the model returned a list of generations, join them
    if isinstance(data, list):
        # sometimes each item is {'generated_text': '...'}
        parts = []
        for item in data:
            if isinstance(item, dict) and "generated_text" in item:
                parts.append(item["generated_text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    # fallback: return stringified json
    return json.dumps(data)


def parse_triples_from_text(text: str):
    # Attempt to parse text as JSON
    try:
        parsed = json.loads(text)
        # Expect a list of objects with subject, predicate, object
        if isinstance(parsed, list):
            triples = []
            for t in parsed:
                if all(k in t for k in ("subject", "predicate", "object")):
                    triples.append((t["subject"], t["predicate"], t["object"]))
            if triples:
                return triples
    except Exception:
        pass
    # Try to extract simple "subject - predicate - object" lines
    triples = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in [" - ", " | ", " -> "]:
            if sep in line:
                parts = [p.strip() for p in line.split(sep)]
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
                    break
    return triples


def triples_to_graph(triples):
    """Convert triples to graph data with entity types and descriptions."""
    nodes = {}
    edges = []
    
    # Entity type detection heuristics
    def infer_type(entity):
        entity_lower = entity.lower()
        words_set = set(re.findall(r'\w+', entity_lower))
        
        # Person indicators - names typically 1-3 words, may have titles
        person_titles = {'dr', 'mr', 'mrs', 'miss', 'prof', 'sir', 'king', 'queen', 'prince', 'princess', 'lord', 'lady', 'ceo', 'president', 'founder', 'director', 'chairman', 'chief'}
        if words_set.intersection(person_titles):
            return "PERSON"
        
        # Organization indicators
        org_words = {'university', 'company', 'corp', 'inc', 'llc', 'ltd', 'government', 'court', 'office', 'agency', 'bank', 'hospital', 'school', 'college', 'institute', 'foundation', 'association'}
        if words_set.intersection(org_words):
            return "ORGANIZATION"
        
        # Legislation indicators
        leg_words = {'act', 'law', 'bill', 'amendment', 'regulation', 'code', 'statute', 'treaty', 'convention', 'protocol'}
        if words_set.intersection(leg_words):
            return "LEGISLATION"
        
        # Government indicators
        gov_words = {'department', 'ministry', 'bureau', 'council', 'committee', 'court', 'senate', 'congress', 'parliament', 'house'}
        if words_set.intersection(gov_words):
            return "GOVERNMENT"
        
        # Legal case indicators
        case_words = {'v', 'vs', 'versus', 'case', 'ruling', 'judgment', 'decision'}
        if words_set.intersection(case_words):
            return "LEGAL_CASE"
        
        # Check if it looks like a person's name (2-3 capitalized words)
        words = entity.split()
        if 1 <= len(words) <= 3 and all(w[0].isupper() for w in words if w):
            # Likely a person name if not matching other categories
            return "PERSON"
        
        # Default to CONCEPT
        return "CONCEPT"
    
    for s, p, o in triples:
        # Add source node
        if s not in nodes:
            node_type = infer_type(s)
            nodes[s] = {
                "id": len(nodes) + 1,
                "label": s,
                "type": node_type,
                "description": f"A {node_type.lower()} mentioned in the text."
            }
        
        # Add target node
        if o not in nodes:
            node_type = infer_type(o)
            nodes[o] = {
                "id": len(nodes) + 1,
                "label": o,
                "type": node_type,
                "description": f"A {node_type.lower()} mentioned in the text."
            }
        
        # Add edge
        edges.append({
            "source": nodes[s]["id"],
            "target": nodes[o]["id"],
            "label": p,
            "description": f"{s} {p} {o}"
        })
    
    return {"nodes": list(nodes.values()), "edges": edges}


def fallback_extract(text: str):
    """Refined extraction logic to avoid poor entities and improve Neo4j-style relationships."""
    import re
    
    # Expanded skip words to avoid "As", "His", "The", etc.
    skip_words = {
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'it', 'its', 'as', 'if', 'then', 'than',
        'so', 'because', 'while', 'when', 'where', 'how', 'what', 'which',
        'who', 'whom', 'whose', 'also', 'more', 'most', 'some', 'any', 'no',
        'not', 'only', 'just', 'about', 'into', 'over', 'after', 'before',
        'between', 'through', 'during', 'under', 'above', 'below', 'up',
        'down', 'out', 'off', 'again', 'further', 'once', 'here', 'there',
        'all', 'each', 'both', 'few', 'other', 'such', 'only', 'own', 'same',
        'too', 'very', 'just', 'now', 'said', 'one', 'two', 'first', 'new',
        'like', 'made', 'make', 'get', 'got', 'come', 'came', 'say', 'see',
        'know', 'take', 'think', 'want', 'use', 'find', 'give', 'tell', 'try',
        'call', 'need', 'feel', 'become', 'leave', 'put', 'keep', 'let',
        'begin', 'seem', 'help', 'show', 'hear', 'play', 'run', 'move', 'live',
        'believe', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
        'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead',
        'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read',
        'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer',
        'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die',
        'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill',
        'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report',
        'decide', 'pull', 'his', 'her', 'their', 'our', 'my', 'your', 'every',
        'each', 'any', 'some', 'no', 'none', 'neither', 'either'
    }
    
    def get_entities(s):
        # Match sequences of capitalized words, but filter out those that start with skip words
        found = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', s)
        results = []
        for e in found:
            words = e.split()
            # If the first word is a common word (like 'As' in 'As Paul'), strip it
            if words[0].lower() in {'as', 'the', 'a', 'an', 'in', 'at', 'his', 'her'}:
                if len(words) > 1:
                    e = " ".join(words[1:])
                else:
                    continue
            
            if e.lower() not in skip_words and len(e) > 2:
                results.append(e)
        return results

    triples = []
    
    # 1. Neo4j-style Relationship Patterns
    patterns = [
        (r'([A-Z][\w\s]+?)\s+(?:was\s+)?founded\s+by\s+([A-Z][\w\s]+)', "FOUNDED_BY"),
        (r'([A-Z][\w\s]+?)\s+(?:is\s+)?headquartered\s+in\s+([A-Z][\w\s]+)', "HEADQUARTERED_IN"),
        (r'([A-Z][\w\s]+?)\s+is\s+(?:a|an)\s+([A-Z][\w\s]+)', "IS_A"),
        (r'([A-Z][\w\s]+?)\s+works\s+(?:at|for)\s+([A-Z][\w\s]+)', "WORKS_AT"),
        (r'([A-Z][\w\s]+?)\s+competes\s+for\s+([A-Z][\w\s]+)', "COMPETES_FOR"),
        (r'([A-Z][\w\s]+?)\s+emerges\s+from\s+([A-Z][\w\s]+)', "ORIGINATES_FROM"),
        (r'([A-Z][\w\s]+?)\s+role\s+in\s+([A-Z][\w\s]+)', "PARTICIPATES_IN"),
        (r'([A-Z][\w\s]+?)\s+follows\s+([A-Z][\w\s]+)', "FOLLOWS"),
        (r'([A-Z][\w\s]+?)\s+focusing\s+on\s+([A-Z][\w\s]+)', "FOCUSES_ON"),
        (r'([A-Z][\w\s]+?)\s+intertwines\s+with\s+([A-Z][\w\s]+)', "INTERTWINES_WITH"),
    ]
    
    for pattern, rel_type in patterns:
        matches = re.findall(pattern, text)
        for s, o in matches:
            triples.append((s.strip(), rel_type, o.strip()))

    # 2. Sentence-level Association (Mesh)
    sentences = re.split(r'[.!?]\s+', text)
    for sent in sentences:
        sent_entities = []
        seen = set()
        for e in get_entities(sent):
            if e not in seen:
                seen.add(e)
                sent_entities.append(e)
        
        if len(sent_entities) >= 2:
            hub = sent_entities[0]
            for other in sent_entities[1:]:
                exists = any((t[0] == hub and t[2] == other) or (t[0] == other and t[2] == hub) for t in triples)
                if not exists:
                    # Look for relationship indicators
                    rel = "RELATED_TO"
                    if "role" in sent.lower(): rel = "PLAYS_ROLE_IN"
                    elif "lives" in sent.lower(): rel = "LIVES_IN"
                    elif "leads" in sent.lower(): rel = "LEADS"
                    elif "controls" in sent.lower(): rel = "CONTROLS"
                    elif "threat" in sent.lower(): rel = "THREATENS"
                    
                    triples.append((hub, rel, other))
    
    # Clean up any bad entities that slipped through
    triples = [(s, r, o) for s, r, o in triples if s.lower() not in skip_words and o.lower() not in skip_words]
    
    return triples


def generate_graph_from_text(text: str) -> dict:
    """
    Generate a knowledge graph from the given text.
    
    Args:
        text: Input text paragraph
        
    Returns:
        Dictionary with 'nodes', 'links', and 'communities' keys
    """
    triples = []
    
    if HF_API and HF_MODEL:
        try:
            out = call_hf_inference(text, HF_MODEL, HF_API)
            triples = parse_triples_from_text(out)
            if not triples:
                print("Model output could not be parsed as triples, falling back to heuristic.")
        except requests.HTTPError as e:
            status = None
            try:
                status = e.response.status_code
            except Exception:
                pass
            if status == 404 and HF_MODEL != FALLBACK_HF_MODEL:
                try:
                    out = call_hf_inference(text, FALLBACK_HF_MODEL, HF_API)
                    triples = parse_triples_from_text(out)
                    if not triples:
                        print("Fallback model output could not be parsed as triples, falling back to heuristic.")
                except Exception as e2:
                    print(f"Fallback model call failed: {e2}. Falling back to heuristic.")
            else:
                print(f"Hugging Face call failed: {e}. Falling back to heuristic.")
        except Exception as e:
            print(f"Hugging Face call failed: {e}. Falling back to heuristic.")

    if not triples:
        triples = fallback_extract(text)

    graph = triples_to_graph(triples)
    
    # Add communities (simple clustering based on connected components)
    communities = compute_communities(graph['nodes'], graph['edges'])
    
    return {
        "nodes": graph['nodes'],
        "links": graph['edges'],
        "communities": communities
    }


def compute_communities(nodes, edges):
    """Simple community detection based on connected components."""
    # Build adjacency list
    adj = {n['id']: [] for n in nodes}
    for e in edges:
        src = e['source'] if isinstance(e['source'], int) else e.get('source', e.get('id'))
        tgt = e['target'] if isinstance(e['target'], int) else e.get('target', e.get('id'))
        if src in adj and tgt in adj:
            adj[src].append(tgt)
            adj[tgt].append(src)
    
    # Find connected components
    visited = set()
    communities = 0
    
    def dfs(node):
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    for node in adj:
        if node not in visited:
            communities += 1
            dfs(node)
    
    return communities


def main():
    parser = argparse.ArgumentParser(description="Generate a simple KG from text using Hugging Face models")
    parser.add_argument("--text", help="The input text (wrap in quotes) ")
    parser.add_argument("--file", help="Path to a text file to read input from")
    parser.add_argument("--out", default="graph_data.json", help="Output JSON file")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Provide --text or --file input. Exiting.")
        sys.exit(1)

    if args.file:
        with open(args.file, "r", encoding="utf8") as f:
            text = f.read()
    else:
        text = args.text

    triples = []
    if HF_API and HF_MODEL:
        try:
            print(f"Calling Hugging Face model '{HF_MODEL}' for triple extraction...")
            out = call_hf_inference(text, HF_MODEL, HF_API)
            triples = parse_triples_from_text(out)
            if not triples:
                print("Model output could not be parsed as triples, falling back to heuristic.")
        except requests.HTTPError as e:
            status = None
            try:
                status = e.response.status_code
            except Exception:
                pass
            if status == 404 and HF_MODEL != FALLBACK_HF_MODEL:
                print(f"Model '{HF_MODEL}' not found (404). Trying fallback model '{FALLBACK_HF_MODEL}'...")
                try:
                    out = call_hf_inference(text, FALLBACK_HF_MODEL, HF_API)
                    triples = parse_triples_from_text(out)
                    if not triples:
                        print("Fallback model output could not be parsed as triples, falling back to heuristic.")
                except Exception as e2:
                    print(f"Fallback model call failed: {e2}. Falling back to heuristic.")
            else:
                print(f"Hugging Face call failed: {e}. Falling back to heuristic.")
        except Exception as e:
            print(f"Hugging Face call failed: {e}. Falling back to heuristic.")

    if not triples:
        triples = fallback_extract(text)

    graph = triples_to_graph(triples)
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"Wrote graph to {args.out} (nodes: {len(graph['nodes'])}, edges: {len(graph['edges'])})")


if __name__ == "__main__":
    main()
