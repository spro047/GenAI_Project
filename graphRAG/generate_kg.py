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
    nodes = {}
    edges = []
    for s, p, o in triples:
        for n in (s, o):
            if n not in nodes:
                nodes[n] = {"id": len(nodes) + 1, "label": n}
        edges.append({"source": nodes[s]["id"], "target": nodes[o]["id"], "label": p})
    return {"nodes": list(nodes.values()), "edges": edges}


def fallback_extract(text: str):
    # Very small heuristic: treat capitalized words and proper nouns as entities
    words = [w.strip(".,;()[]") for w in text.split()]
    candidates = [w for w in words if w and w[0].isupper()]
    # pair consecutive candidates as simple relations
    triples = []
    for i in range(len(candidates) - 1):
        triples.append((candidates[i], "related_to", candidates[i + 1]))
    return triples


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
