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
from serpapi import GoogleSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


HF_API = os.getenv("HUGGINGFACE_API_KEY", "")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
# Local LLM settings
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/v1/chat/completions")

# Local GGUF settings (direct file loading)
USE_LOCAL_GGUF = os.getenv("USE_LOCAL_GGUF", "false").lower() == "true"
LOCAL_GGUF_MODEL = os.getenv("LOCAL_GGUF_MODEL")

# Global variable to cache the local model instance
_LOCAL_MODEL_INSTANCE = None

# Fallback public model to try if the configured model path is not found
FALLBACK_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def call_local_llm(text: str) -> str:
    """Calls a local LLM server (OpenAI-compatible like llama.cpp or Ollama)."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a knowledge graph extractor. Your task is to identify entities and their relationships from text. Output ONLY a raw JSON list of objects with keys 'subject', 'predicate', and 'object'. No talk, no markdown blocks."
            },
            {
                "role": "user",
                "content": f"Extract triples from this text:\n\n{text}"
            }
        ],
        "temperature": 0.1
    }
    try:
        resp = requests.post(LOCAL_LLM_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if 'choices' in data:
            return data['choices'][0]['message']['content']
        return str(data)
    except Exception as e:
        print(f"Local LLM call failed: {e}")
        return ""


def call_local_gguf(text: str) -> str:
    """Calls a local GGUF model directly using llama-cpp-python."""
    global _LOCAL_MODEL_INSTANCE
    
    if not LOCAL_GGUF_MODEL or not os.path.exists(LOCAL_GGUF_MODEL):
        print(f"Local GGUF model path not found: {LOCAL_GGUF_MODEL}")
        return ""
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. Cannot use local GGUF model.")
        return ""
    
    if _LOCAL_MODEL_INSTANCE is None:
        print(f"Loading local GGUF model: {LOCAL_GGUF_MODEL}...")
        try:
            _LOCAL_MODEL_INSTANCE = Llama(
                model_path=LOCAL_GGUF_MODEL,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load GGUF model: {e}")
            return ""

    prompt = (
        "<s>[INST] You are a knowledge graph extractor. Extract entities and their relationships from the following text. "
        "Output ONLY a raw JSON list of objects with keys 'subject', 'predicate', and 'object'. "
        "Do not include any explanation or markdown formatting.\n\n"
        f"Text: {text} [/INST]"
    )
    
    try:
        output = _LOCAL_MODEL_INSTANCE(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["</s>"]
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Local GGUF inference failed: {e}")
        return ""


def search_serpapi(query: str) -> str:
    """Searches SerpAPI for additional information about a query."""
    if not SERPAPI_KEY:
        print("DEBUG: SERPAPI_KEY not found in environment.")
        return ""
    
    print(f"DEBUG: Searching SerpAPI for: {query}...")
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 3
        })
        results = search.get_dict()
        
        snippets = []
        if "organic_results" in results:
            for res in results["organic_results"][:3]:
                if "snippet" in res:
                    snippets.append(res["snippet"])
        
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            if "description" in kg:
                snippets.append(f"Knowledge Graph: {kg['description']}")
                
        return "\n".join(snippets)
    except Exception as e:
        print(f"SerpAPI search failed: {e}")
        return ""


def extract_key_entities(text: str) -> list:
    """Extracts potential key entities for searching."""
    # Simple heuristic: capitalized words that aren't at the start of sentences (mostly)
    # or just sequences of capitalized words.
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Filter out common stop words if they got caught
    stop_words = {"The", "A", "An", "In", "On", "At", "To", "For", "Of", "With", "By"}
    unique_entities = list(set([e for e in entities if e not in stop_words]))
    return unique_entities[:3] # Limit to top 3 for searching


def get_augmented_text(text: str) -> str:
    """Augments the input text with information from SerpAPI."""
    entities = extract_key_entities(text)
    if not entities:
        return text
    
    additional_info = []
    for entity in entities:
        info = search_serpapi(entity)
        if info:
            additional_info.append(f"Information about {entity}:\n{info}")
    
    if not additional_info:
        return text
        
    augmented = text + "\n\n### ADDITIONAL KNOWLEDGE FROM SEARCH ###\n" + "\n\n".join(additional_info)
    return augmented


def call_hf_inference(text: str, model: str, token: str) -> str:
    from huggingface_hub import InferenceClient
    
    prompt = f"""You are an expert Knowledge Graph Engineer specializing in information extraction and semantic structuring. Your task is to construct a high-density, high-precision Knowledge Graph from the provided text, ensuring maximum completeness, correctness, and consistency. The objective is to extract all meaningful, atomic relationships between entities while maintaining a high level of accuracy and semantic clarity.

Return only a raw JSON array without any markdown, explanation, or additional text. Each entry must strictly follow this schema: {{"subject": "Entity A", "predicate": "RELATIONSHIP_TYPE", "object": "Entity B"}}. Ensure the output is valid JSON with no trailing commas or formatting errors.

All entities must be normalized using the most complete and canonical form available in the text. Resolve all coreferences such as pronouns (“he”, “she”, “it”, “they”) to their correct entities and maintain consistent naming across all triples. Avoid duplication caused by aliases, abbreviations, or partial names.

Predicates must be dynamically derived from the text to capture the precise semantic meaning of the interaction. While relationships like "FOUNDED_BY" or "LOCATED_IN" are common, you should define the most descriptive and accurate predicate based on the specific context of the sentence. Predicates must be written in uppercase and standardized across the document (e.g., use the same predicate for the same type of relationship). Avoid vague predicates like "IS", "HAS", or "RELATED_TO".

Each triple must be atomic and represent exactly one fact. Do not combine multiple relationships into a single triple. Extract relationships exhaustively, including primary facts, secondary details, implicit relationships, and contextual links such as temporal, spatial, organizational, and functional connections.

Avoid duplicate triples and merge semantically equivalent relationships into a single standardized representation. Ensure all extracted facts are grounded in the provided text or are clearly inferable from it. Do not hallucinate or introduce unsupported information. If a relationship is uncertain, omit it.

Preserve temporal and numerical information accurately. Represent dates, quantities, and values explicitly using appropriate predicates such as "FOUNDED_IN", "BORN_ON", or "HAS_VALUE". Internally infer entity types such as Person, Organization, Location, or Event to improve relationship accuracy, but do not include these types in the output unless explicitly mentioned in the text.

If additional knowledge is provided, use it carefully to validate relationships and enhance completeness, but do not introduce external facts that are not strongly supported. Maintain strict output cleanliness and ensure the final response is a valid, well-structured JSON array.

TEXT TO PROCESS:
{text}

JSON OUTPUT:
"""
    client = InferenceClient(token=token)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=1500
    )
    return response.choices[0].message.content


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


def triples_to_graph(triples, text=""):
    """Convert triples to graph data with entity types and descriptions.
    Uses surrounding text context for dynamic type inference if available."""
    nodes = {}
    edges = []
    
    # Entity merging cache
    _merged_entities = {}

    def normalize_entity(name):
        name = name.strip()
        
        # Merge fragments into full names (e.g. "Musk" -> "Elon Musk")
        for full_name in _merged_entities:
            if name != full_name and (name in full_name or full_name in name):
                # If name is a fragment of an already seen full name, return full name
                if len(name) < len(full_name):
                    return full_name
        
        # Basic cleanup: remove leading 'The ' if it's not part of a formal name
        if name.lower().startswith("the "):
            test_name = name[4:]
            if test_name and test_name[0].isupper():
                return name 
            return test_name
        
        # Register new full name
        if len(name.split()) >= 2:
            _merged_entities[name] = True
            
        return name

    # Entity type detection heuristics (dynamic + fallback)
    def infer_type(entity, text):
        entity_lower = entity.lower()
        
        # 1. Dynamic Type Inference from Context
        if text:
            escaped_entity = re.escape(entity)
            # Patterns: "Entity is a/an Type", "Entity, a/an Type"
            # We restrict to a/an to avoid catching adjectives like "the young Paul"
            patterns = [
                rf"{escaped_entity}(?:\s*,\s*|\s+is\s+|\s+was\s+)(?:a|an)\s+([a-zA-Z\-]+(?:\s+[a-zA-Z\-]+){{0,2}}?)(?=\s+who|\s+that|\s+which|\s+where|\s+when|,|\.|;|:|$)"
            ]
            stop_words = {'very', 'much', 'only', 'just', 'new', 'old', 'good', 'bad', 'great', 'small', 'large', 'big', 'main', 'primary', 'secondary', 'first', 'last', 'of', 'for', 'with'}
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    phrase = match.group(1).strip()
                    words = [w for w in phrase.split() if w.lower() not in stop_words]
                    if words:
                        dynamic_type = "_".join(words[-2:]).upper()
                        if 2 < len(dynamic_type) < 30:
                            return dynamic_type

        # 2. Fallback Heuristics
        words_set = set(re.findall(r'\w+', entity_lower))
        
        # Person indicators - names typically 1-3 words, may have titles
        person_titles = {'dr', 'mr', 'mrs', 'miss', 'prof', 'sir', 'king', 'queen', 'prince', 'princess', 'lord', 'lady', 'ceo', 'president', 'founder', 'director', 'chairman', 'chief'}
        if words_set.intersection(person_titles):
            return "PERSON"
        
        # Organization indicators
        org_words = {'university', 'company', 'corp', 'inc', 'llc', 'ltd', 'government', 'court', 'office', 'agency', 'bank', 'hospital', 'school', 'college', 'institute', 'foundation', 'association', 'spacex', 'tesla', 'google', 'microsoft', 'apple'}
        if words_set.intersection(org_words):
            return "ORGANIZATION"
        
        # Location indicators
        loc_words = {'city', 'state', 'country', 'island', 'mountain', 'river', 'ocean', 'sea', 'continent', 'africa', 'america', 'europe', 'asia', 'south', 'north', 'east', 'west', 'arrakis', 'dune', 'mars', 'earth', 'london', 'paris', 'tokyo', 'u.s.a.', 'u.k.', 'e.u.'}
        if words_set.intersection(loc_words) or entity_lower in {'u.s.a.', 'u.k.', 'e.u.', 'usa', 'uk'}:
            return "LOCATION"

        # Legislation indicators
        leg_words = {'act', 'law', 'bill', 'amendment', 'regulation', 'code', 'statute', 'treaty', 'convention', 'protocol'}
        if words_set.intersection(leg_words):
            return "LEGISLATION"
        
        # Government indicators
        gov_words = {'department', 'ministry', 'bureau', 'council', 'committee', 'court', 'senate', 'congress', 'parliament', 'house', 'fremen', 'empire'}
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
    
    for s_raw, p, o_raw in triples:
        s = normalize_entity(s_raw)
        o = normalize_entity(o_raw)
        # Add source node
        if s not in nodes:
            node_type = infer_type(s, text)
            nodes[s] = {
                "id": len(nodes) + 1,
                "label": s,
                "type": node_type,
                "description": f"A {node_type.replace('_', ' ').lower()} mentioned in the text."
            }
        
        # Add target node
        if o not in nodes:
            node_type = infer_type(o, text)
            nodes[o] = {
                "id": len(nodes) + 1,
                "label": o,
                "type": node_type,
                "description": f"A {node_type.replace('_', ' ').lower()} mentioned in the text."
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
        'each', 'any', 'some', 'no', 'none', 'neither', 'either', 'he', 'she',
        'they', 'we', 'you', 'it', 'its', 'shortly', 'while', 'as', 'duke'
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
    
    # 1. Neo4j-style Relationship Patterns (Direct extraction)
    patterns = [
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was\s+)?founded\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "FOUNDED_BY"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?headquartered\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "HEADQUARTERED_IN"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "IS_A"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works\s+(?:at|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "WORKS_AT"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+competes\s+with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "COMPETES_WITH"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+born\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "BORN_IN"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+lives\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "LIVES_IN"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+role\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "PARTICIPATES_IN"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+follows\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "FOLLOWS"),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+focusing\s+on\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "FOCUSES_ON"),
    ]
    
    for pattern, rel_type in patterns:
        matches = re.findall(pattern, text)
        for s, o in matches:
            triples.append((s.strip(), rel_type, o.strip()))

    # 2. Sentence-level Association (Mesh) - Enhanced with basic coreference
    sentences = re.split(r'[.!?]\s+', text)
    last_person = None
    
    for sent in sentences:
        # Find all sequences of capitalized words, allowing 'of', 'the' in middle, and consecutive caps
        full_names = re.findall(r'\b[A-Z][A-Za-z\']+(?:\s+(?:of|the)\s+[A-Z][A-Za-z\']+|\s+[A-Z][A-Za-z\']+)*\b', sent)
        
        valid_entities = []
        for name in full_names:
            words = name.split()
            # Strip common starting words if capitalized at beginning of sentence
            if words[0].lower() in {'the', 'a', 'an', 'in', 'at', 'his', 'her', 'as', 'they', 'shortly', 'while', 'but', 'and', 'or', 'so'}:
                if len(words) > 1 and words[1][0].isupper():
                    name = " ".join(words[1:])
                else:
                    continue
            
            if name.lower() not in skip_words and len(name) > 2:
                valid_entities.append(name)
        
        sent_entities = []
        seen = set()
        for e in valid_entities:
            if e not in seen:
                seen.add(e)
                sent_entities.append(e)
        
        # Simple Coreference: Replace pronouns with last_person
        pronouns = re.findall(r'\b(He|She|It|They)\b', sent)
        if pronouns and last_person:
            for p in pronouns:
                if last_person not in sent_entities:
                    sent_entities.insert(0, last_person)
        
        # Update last_person for next sentence
        for e in sent_entities:
            # If entity looks like a name (capitalized words), set as last_person
            if any(title in e.lower() for title in {'mr', 'mrs', 'ms', 'dr', 'prof'}) or len(e.split()) >= 2:
                last_person = e
                break

        if len(sent_entities) >= 2:
            # Connect the first entity to all others (hub and spoke for the sentence)
            hub = sent_entities[0]
            for other in sent_entities[1:]:
                exists = any((t[0] == hub and t[2] == other) or (t[0] == other and t[2] == hub) for t in triples)
                if not exists:
                    # Dynamic Relationship Extraction: Find the text between the two entities
                    # If they appear in the same sentence, try to extract the middle part
                    escaped_hub = re.escape(hub)
                    escaped_other = re.escape(other)
                    pattern = f"{escaped_hub}(.*?){escaped_other}"
                    match = re.search(pattern, sent, re.IGNORECASE)
                    
                    rel = "ASSOCIATED_WITH"
                    if match:
                        mid = match.group(1).strip()
                        # Remove common filler words and punctuation
                        mid = re.sub(r'[^\w\s]', '', mid)
                        mid = re.sub(r'\b(the|a|an|is|are|was|were|has|have|had|been|to|in|at|on|for|with|by|from|and|but|or|their|his|her)\b', '', mid, flags=re.IGNORECASE).strip()
                        # If we have a verb or meaningful phrase, use it (max 2 words)
                        if mid and len(mid.split()) <= 2:
                            rel = mid.upper().replace(' ', '_')
                        else:
                            # Fallback to keyword search if mid is too complex
                            lower_sent = sent.lower()
                            if re.search(r'\b(founded|created|established|started)\b', lower_sent): rel = "FOUNDED"
                            elif re.search(r'\b(lives|resides|dwells)\b', lower_sent): rel = "LIVES_IN"
                            elif re.search(r'\b(works|employed|ceo|founder|director)\b', lower_sent): rel = "WORKS_AT"
                            elif re.search(r'\b(born|birth)\b', lower_sent): rel = "BORN_IN"
                            elif re.search(r'\b(leads|heads|manages|controls)\b', lower_sent): rel = "LEADS"
                            elif re.search(r'\b(located|based|situated)\b', lower_sent): rel = "LOCATED_IN"
                    
                    triples.append((hub, rel, other))
    
    return triples


def generate_graph_from_text(text: str) -> dict:
    """
    Generate a knowledge graph from the given text.
    
    Args:
        text: Input text paragraph
        
    Returns:
        Dictionary with 'nodes', 'links', 'communities', and metadata keys
    """
    triples = []
    extraction_method = "Heuristic Fallback"
    search_augmented = False
    
    # 1. Augment text with SerpAPI knowledge
    augmented_text = get_augmented_text(text)
    if len(augmented_text) > len(text):
        search_augmented = True
        print("Text augmented with SerpAPI knowledge.")
    
    # 2. Extract triples using LLM (with fallback to heuristic)
    if USE_LOCAL_GGUF:
        print(f"Using local GGUF model: {LOCAL_GGUF_MODEL}...")
        out = call_local_gguf(augmented_text)
        triples = parse_triples_from_text(out)
        if triples: extraction_method = f"Local GGUF ({LOCAL_GGUF_MODEL})"
    
    if not triples and USE_LOCAL_LLM:
        print(f"Using Local LLM at {LOCAL_LLM_URL}...")
        out = call_local_llm(augmented_text)
        triples = parse_triples_from_text(out)
        if triples: extraction_method = "Local LLM"
    
    if not triples and HF_API and HF_MODEL:
        try:
            print(f"Using Hugging Face model: {HF_MODEL}...")
            out = call_hf_inference(augmented_text, HF_MODEL, HF_API)
            
            # --- DEBUG LOGGING ---
            print("\n" + "="*50)
            print("RAW LLM OUTPUT:")
            print(out)
            print("="*50 + "\n")
            with open("raw_llm_output.txt", "w", encoding="utf-8") as f:
                f.write(out)
            # ---------------------
            
            triples = parse_triples_from_text(out)
            if triples: 
                extraction_method = f"Hugging Face ({HF_MODEL})"
            else:
                print("Model output could not be parsed as triples, falling back to heuristic.")
        except Exception as e:
            # Check if it's a model not supported error to try fallback
            if "model_not_supported" in str(e) and HF_MODEL != FALLBACK_HF_MODEL:
                try:
                    print(f"Model '{HF_MODEL}' not supported. Trying fallback model '{FALLBACK_HF_MODEL}'...")
                    out = call_hf_inference(text, FALLBACK_HF_MODEL, HF_API)
                    triples = parse_triples_from_text(out)
                    if not triples:
                        print("Fallback model output could not be parsed as triples, falling back to heuristic.")
                except Exception as e2:
                    print(f"Fallback model call failed: {e2}. Falling back to heuristic.")
            else:
                print(f"Hugging Face call failed: {e}. Falling back to heuristic.")

    if not triples:
        triples = fallback_extract(text)

    graph = triples_to_graph(triples, augmented_text)
    
    # Add communities (simple clustering based on connected components)
    communities = compute_communities(graph['nodes'], graph['edges'])
    
    return {
        "nodes": graph['nodes'],
        "links": graph['edges'],
        "communities": communities,
        "extraction_method": extraction_method,
        "search_augmented": search_augmented
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


def query_graph(query: str, nodes: list, links: list) -> str:
    """A simple query system to find answers in the graph data."""
    query = query.lower()
    
    # Helper to get node by ID
    def get_node(nid):
        return next((n for n in nodes if n['id'] == nid), None)

    # 1. Search for direct entity matches
    for link in links:
        # link might have source/target as IDs or objects (from D3 simulation)
        s_val = link['source']
        t_val = link['target']
        
        src_id = s_val['id'] if isinstance(s_val, dict) else s_val
        tgt_id = t_val['id'] if isinstance(t_val, dict) else t_val
        
        src_node = get_node(src_id)
        tgt_node = get_node(tgt_id)
        
        if not src_node or not tgt_node: continue
        
        s = src_node['label'].lower()
        t = tgt_node['label'].lower()
        p = link['label'].lower()
        
        # Check if query mentions the subject/object AND relationship
        # Or if it's a "who/what" question about one of the entities
        if (s in query or t in query):
            # Check for relationship match or question words
            if p in query or any(w in query for w in ["who", "what", "where", "how", "tell", "show"]):
                return f"According to the graph, {src_node['label']} {link['label']} {tgt_node['label']}."
            
    return "I couldn't find a direct answer in the current graph. Try extracting more text or refining your query."


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
    if USE_LOCAL_GGUF:
        print(f"Using local GGUF model: {LOCAL_GGUF_MODEL}...")
        out = call_local_gguf(text)
        triples = parse_triples_from_text(out)

    if not triples and USE_LOCAL_LLM:
        print(f"Using local LLM at {LOCAL_LLM_URL}...")
        out = call_local_llm(text)
        triples = parse_triples_from_text(out)

    if not triples and HF_API and HF_MODEL:
        try:
            print(f"Calling Hugging Face model '{HF_MODEL}' for triple extraction...")
            out = call_hf_inference(text, HF_MODEL, HF_API)
            
            # --- DEBUG LOGGING ---
            print("\n" + "="*50)
            print("RAW LLM OUTPUT:")
            print(out)
            print("="*50 + "\n")
            with open("raw_llm_output.txt", "w", encoding="utf-8") as f:
                f.write(out)
            # ---------------------
            
            triples = parse_triples_from_text(out)
            if not triples:
                print("Model output could not be parsed as triples, falling back to heuristic.")
        except Exception as e:
            if "model_not_supported" in str(e) and HF_MODEL != FALLBACK_HF_MODEL:
                print(f"Model '{HF_MODEL}' not supported. Trying fallback model '{FALLBACK_HF_MODEL}'...")
                try:
                    out = call_hf_inference(text, FALLBACK_HF_MODEL, HF_API)
                    triples = parse_triples_from_text(out)
                    if not triples:
                        print("Fallback model output could not be parsed as triples, falling back to heuristic.")
                except Exception as e2:
                    print(f"Fallback model call failed: {e2}. Falling back to heuristic.")
            else:
                print(f"Hugging Face call failed: {e}. Falling back to heuristic.")

    if not triples:
        triples = fallback_extract(text)

    graph = triples_to_graph(triples, text)
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"Wrote graph to {args.out} (nodes: {len(graph['nodes'])}, edges: {len(graph['edges'])})")


if __name__ == "__main__":
    main()
