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
    
    prompt = (
        "You are a precision Knowledge Graph engine. Return ONLY a raw JSON array, no markdown, no explanation.\n\n"
        "Each item: {\"subject\": \"Entity\", \"predicate\": \"RELATION\", \"object\": \"Entity\"}\n\n"
        "RULES:\n"
        "1. Entity names = clean proper nouns only. NEVER put a job title inside an entity name.\n"
        "   BAD: {\"subject\":\"Sundar Pichai\",\"predicate\":\"WORKS_AT\",\"object\":\"CEO of Google\"}\n"
        "   GOOD: {\"subject\":\"Google\",\"predicate\":\"CEO\",\"object\":\"Sundar Pichai\"}\n"
        "2. Predicate = exact role or relationship, NEVER a generic verb:\n"
        "   Roles   -> CEO, CTO, CFO, FOUNDER, CO_FOUNDER, CHAIRMAN, CHIEF_AI_SCIENTIST\n"
        "   Creation-> CREATED, DEVELOPED, BUILT, LAUNCHED\n"
        "   Money   -> INVESTED_IN, ACQUIRED, FUNDED, PARTNERED_WITH\n"
        "   Location-> HEADQUARTERED_IN, BASED_IN, BORN_IN\n"
        "   Compete -> COMPETES_WITH\n"
        "   Power   -> POWERED_BY, RUNS_ON, REPLACED\n"
        "   Org     -> OWNED_BY, SUBSIDIARY_OF, LEADS\n"
        "   BANNED  : WORKS_AT, LIVES_AT, IS_A, HAS, IS, RELATED_TO, ASSOCIATED_WITH\n"
        "3. Direction: Company--CEO-->Person, Person--FOUNDED-->Company, Company--CREATED-->Product\n"
        "4. Resolve pronouns. One atomic fact per triple. No duplicates.\n\n"
        "EXAMPLES:\n"
        "[\n"
        "  {\"subject\":\"Tesla\",\"predicate\":\"CEO\",\"object\":\"Elon Musk\"},\n"
        "  {\"subject\":\"Elon Musk\",\"predicate\":\"FOUNDED\",\"object\":\"SpaceX\"},\n"
        "  {\"subject\":\"Google\",\"predicate\":\"CEO\",\"object\":\"Sundar Pichai\"},\n"
        "  {\"subject\":\"Google\",\"predicate\":\"CREATED\",\"object\":\"Gemini\"},\n"
        "  {\"subject\":\"Microsoft\",\"predicate\":\"INVESTED_IN\",\"object\":\"OpenAI\"},\n"
        "  {\"subject\":\"SpaceX\",\"predicate\":\"HEADQUARTERED_IN\",\"object\":\"Hawthorne\"}\n"
        "]\n\n"
        f"TEXT:\n{text}\n\nJSON OUTPUT (array only):"
    )
    client = InferenceClient(token=token)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=1500
    )
    return response.choices[0].message.content


def parse_triples_from_text(text: str):
    # Strip markdown code fences
    clean = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    # Find JSON array anywhere in output
    match = re.search(r'\[.*\]', clean, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                triples = []
                for t in parsed:
                    if all(k in t for k in ("subject", "predicate", "object")):
                        triples.append((t["subject"].strip(), t["predicate"].strip(), t["object"].strip()))
                if triples:
                    return triples
        except Exception:
            pass
    # Line-based fallback
    triples = []
    for line in clean.splitlines():
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
    
    # 1. Alias Mapping
    alias_map = {
        "belly": "Isabel Conklin",
        "isabel": "Isabel Conklin"
    } 
    
    alias_predicates = {'KNOWN_AS', 'ALSO_KNOWN_AS', 'ALIAS'}
    for s, p, o in triples:
        if p.upper() in alias_predicates:
            alias_map[s.strip().lower()] = o.strip()

    def resolve_alias(name):
        return alias_map.get(name.lower(), name)

    # Entity merging cache
    _merged_entities = {}

    def normalize_entity(name):
        name = name.strip()
        
        # Resolve aliases first
        canonical = resolve_alias(name)
        if canonical != name:
            return canonical
        
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

    def infer_type(entity, text):
        el = entity.lower()
        words = set(re.findall(r'\w+', el))

        # Person indicators
        person_titles = {'dr','mr','mrs','miss','prof','sir','king','queen','prince','princess','lord','lady','ceo','president','founder','director','chairman','chief','scientist','researcher','engineer'}
        if words & person_titles:
            return "PERSON"

        # AI Model indicators (check before ORGANIZATION)
        ai_models = {'gpt','gpt-4','gpt-3','chatgpt','gemini','bard','claude','llama','llama3','alphacode','alphastar','alphago','copilot','dall-e','sora','mistral','falcon','bloom','palm'}
        if words & ai_models or any(m in el for m in ['gpt-', 'llama-', 'claude-']):
            return "AI_MODEL"

        # Product / Technology indicators
        product_words = {'gpu','chip','processor','hardware','software','api','platform','framework','tool','library','model','network','system','robot','satellite','rocket','car','vehicle','phone','device'}
        if words & product_words:
            return "PRODUCT"

        # Organization indicators
        org_words = {'university','company','corp','inc','llc','ltd','government','court','office','agency','bank','hospital','school','college','institute','foundation','association','lab','laboratory','openai','spacex','tesla','google','microsoft','apple','anthropic','nvidia','meta','amazon','deepmind','hugging'}
        if words & org_words:
            return "ORGANIZATION"

        # Location indicators
        loc_words = {'city','state','country','island','mountain','river','ocean','sea','continent','africa','america','europe','asia','south','north','east','west','mars','earth','london','paris','tokyo','california','washington','seattle','redmond','menlo','hawthorne','santa','san','new','york','boston'}
        if words & loc_words:
            return "LOCATION"

        # Event indicators
        event_words = {'war','battle','conference','summit','election','tournament','championship','match','game','competition','award','ceremony','trial','hearing','launch','debut'}
        if words & event_words:
            return "EVENT"

        # Legislation indicators
        leg_words = {'act','law','bill','amendment','regulation','code','statute','treaty','convention','protocol'}
        if words & leg_words:
            return "LEGISLATION"

        # Government indicators
        gov_words = {'department','ministry','bureau','council','committee','senate','congress','parliament','house','court'}
        if words & gov_words:
            return "GOVERNMENT"

        # Technology concept
        tech_words = {'ai','machine','learning','deep','neural','algorithm','data','cloud','computing','blockchain','crypto','quantum'}
        if words & tech_words:
            return "TECHNOLOGY"

        # 1-3 capitalized words Ã¢â€ â€™ likely a person
        ws = entity.split()
        if 1 <= len(ws) <= 3 and all(w[0].isupper() for w in ws if w):
            return "PERSON"

        return "CONCEPT"
    
    for s_raw, p_raw, o_raw in triples:
        p = p_raw.upper()
        if p in alias_predicates:
            # Skip raw alias edges so they don't appear in the graph
            continue
            
        s = normalize_entity(s_raw)
        o = normalize_entity(o_raw)
        
        if s == o:
            # Do NOT create relationships between aliases of the same entity
            continue
            
        # Add source node
        if s not in nodes:
            node_type = infer_type(s, text)
            nodes[s] = {
                "id": len(nodes) + 1,
                "label": s,
                "type": node_type,
                "description": f"A {node_type.replace('_', ' ').lower()} mentioned in the text.",
                "aliases": []
            }
        
        # Add target node
        if o not in nodes:
            node_type = infer_type(o, text)
            nodes[o] = {
                "id": len(nodes) + 1,
                "label": o,
                "type": node_type,
                "description": f"A {node_type.replace('_', ' ').lower()} mentioned in the text.",
                "aliases": []
            }
            
        # Record aliases if raw name differs from canonical
        s_alias = s_raw.strip()
        if s_alias != s and s_alias not in nodes[s]["aliases"]:
            nodes[s]["aliases"].append(s_alias)
            
        o_alias = o_raw.strip()
        if o_alias != o and o_alias not in nodes[o]["aliases"]:
            nodes[o]["aliases"].append(o_alias)
        
        # Deduplicate exactly identical relationships before adding
        s_id = nodes[s]["id"]
        o_id = nodes[o]["id"]
        
        is_dup = any(e for e in edges if e["source"] == s_id and e["target"] == o_id and e["label"] == str(p_raw))
        if is_dup:
            continue
            
        edges.append({
            "source": s_id,
            "target": o_id,
            "label": str(p_raw),
            "description": f"{s} {p_raw} {o}"
        })
    
    return {"nodes": list(nodes.values()), "edges": edges}


def fallback_extract(text: str):
    """
    Role-based heuristic extractor.
    Uses strict case-sensitive entity patterns so that lowercase words like
    'is', 'was', 'which', 'by', 'the' are NEVER captured as entity names.
    """
    # Entity: starts with uppercase, each subsequent word also uppercase (no IGNORECASE)
    E = r'([A-Z][A-Za-z0-9]*(?:[ \-][A-Z][A-Za-z0-9]*)*)'

    LEADING_STRIP = {'by','the','a','an','both','which','that','their','its',
                     'this','these','those','all','is','was','are','were','named'}
    TRAILING_STRIP = {'is','was','are','were','be','been','the','a','an','and','or','named'}
    JUNK = {'which','who','that','what','where','when','how','this','these','those',
            'both','all','some','any','is','was','are','were','be','been','being',
            'has','have','had','do','does','did','the','a','an','and','or','but',
            'for','of','in','on','at','to','by','from','with','it','its','they',
            'them','their','he','she','him','her','we','us','our','you','i'}

    def clean(name):
        words = name.strip().split()
        while words and words[0].lower() in LEADING_STRIP:
            words = words[1:]
        while words and words[-1].lower() in TRAILING_STRIP:
            words = words[:-1]
        return ' '.join(words)

    def valid(name):
        if not name or len(name) < 2:
            return False
        if not name[0].isupper():
            return False
        if name.lower() in JUNK:
            return False
        if all(w.lower() in JUNK for w in name.split()):
            return False
        return True

    def split_founders(raw):
        raw = re.sub(r',?\s+and\s+', '|', raw, flags=re.IGNORECASE)
        raw = re.sub(r',\s*', '|', raw)
        return [p.strip() for p in raw.split('|') if p.strip()]

    triples = []
    seen = set()

    def add(s, p, o):
        s, o = clean(s), clean(o)
        if not valid(s) or not valid(o) or s.lower() == o.lower():
            return
        key = (s.lower(), p, o.lower())
        if key not in seen:
            seen.add(key)
            triples.append((s, p, o))

    role_pat = (r'CEO|CTO|CFO|COO|CMO|President|Chairman|Director|'
                r'Founder|Co-Founder|'
                r'Chief [A-Z][a-z]+ (?:Officer|Scientist|Architect)')

    # Role: "Person is/serves as the ROLE of Company" -> (Company, ROLE, Person)
    for m in re.finditer(
        rf'{E}\s+(?:is|serves\s+as|was|became)\s+(?:the\s+)?({role_pat})\s+(?:of|at|for)\s+{E}',
        text
    ):
        role = re.sub(r'\s+', '_', m.group(2).strip()).upper()
        add(m.group(3), role, m.group(1))

    # Founded by (handles "by A, B, and C")
    for m in re.finditer(
        rf'{E}\s+(?:was\s+)?founded\s+by\s+([A-Z][^.]+?)(?=\.|,\s+both|\s+to\s+|\s+who\s+|$)',
        text
    ):
        company = clean(m.group(1))
        for founder in split_founders(m.group(2)):
            f = clean(founder)
            if valid(f):
                add(company, 'FOUNDED_BY', f)

    # Leads / Heads
    for m in re.finditer(rf'{E}\s+(?:leads|heads|manages|directs)\s+{E}', text):
        add(m.group(1), 'LEADS', m.group(2))

    # Creation
    for m in re.finditer(
        rf'{E}\s+(?:created|developed|built|designed|launched|released|introduced)\s+(?:the\s+)?{E}',
        text
    ):
        add(m.group(1), 'CREATED', m.group(2))

    # Powered by
    for m in re.finditer(rf'{E}\s+is\s+powered\s+by\s+(?:the\s+)?{E}', text):
        add(m.group(1), 'POWERED_BY', m.group(2))

    # Replaced
    for m in re.finditer(
        rf'{E}\s+replaced\s+(?:their\s+earlier\s+model\s+named\s+|their\s+)?{E}',
        text
    ):
        add(m.group(1), 'REPLACED', m.group(2))

    # Invested in
    for m in re.finditer(rf'{E}\s+invested\s+(?:heavily\s+)?(?:in|into)\s+{E}', text):
        add(m.group(1), 'INVESTED_IN', m.group(2))

    # Acquired
    for m in re.finditer(rf'{E}\s+(?:acquired|bought)\s+{E}', text):
        add(m.group(1), 'ACQUIRED', m.group(2))

    # Partnered
    for m in re.finditer(rf'{E}\s+partnered\s+with\s+{E}', text):
        add(m.group(1), 'PARTNERED_WITH', m.group(2))

    # Owned by
    for m in re.finditer(rf'{E}\s+(?:is\s+)?owned\s+by\s+{E}', text):
        add(m.group(1), 'OWNED_BY', m.group(2))

    # Owns
    for m in re.finditer(rf'{E}\s+(?:owns|controls)\s+{E}', text):
        add(m.group(1), 'OWNS', m.group(2))

    # Headquartered in
    for m in re.finditer(rf'{E},?\s+(?:is\s+)?headquartered\s+in\s+{E}', text):
        add(m.group(1), 'HEADQUARTERED_IN', m.group(2))

    # Based in / located in
    for m in re.finditer(rf'{E}\s+(?:is\s+)?(?:based|located)\s+in\s+{E}', text):
        add(m.group(1), 'BASED_IN', m.group(2))

    # Born in
    for m in re.finditer(rf'{E}\s+(?:was\s+)?born\s+in\s+{E}', text):
        add(m.group(1), 'BORN_IN', m.group(2))

    # Competes with
    for m in re.finditer(rf'{E}\s+competes\s+(?:directly\s+)?with\s+{E}', text):
        add(m.group(1), 'COMPETES_WITH', m.group(2))

    # Defeated
    for m in re.finditer(rf'{E}\s+(?:defeated|beat)\s+{E}', text):
        add(m.group(1), 'DEFEATED', m.group(2))

    # Previously worked at
    for m in re.finditer(
        rf'{E}\s+(?:previously|formerly)\s+worked\s+(?:at|for)\s+{E}',
        text
    ):
        add(m.group(1), 'FORMERLY_AT', m.group(2))

    # Provides X with Y
    for m in re.finditer(rf'{E}\s+provides\s+\S+\s+with\s+{E}', text):
        add(m.group(1), 'PROVIDES', m.group(2))


    # â”€â”€ Narrative / story patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # "X ruled/governs/governed by Y" -> (X, RULED_BY, Y)
    for m in re.finditer(rf'{E}\s+(?:is\s+a\s+\S+\s+family\s+)?ruled\s+by\s+{E}', text):
        add(m.group(1), 'RULED_BY', m.group(2))

    # "X is the head/leader/ruler of Y" -> (Y, LEADER, X)  or  "X is the leader of Y"
    for m in re.finditer(
        rf'{E}\s+is\s+(?:the\s+)?(head|leader|ruler|chief|commander|lord)\s+of\s+{E}',
        text
    ):
        role = m.group(2).upper()
        add(m.group(3), role, m.group(1))

    # "X served as the ROLE of Y" -> (Y, ROLE, X)
    for m in re.finditer(
        rf'{E}\s+served\s+as\s+(?:the\s+)?([A-Za-z][a-z]+(?:\s+[a-z]+)?)\s+of\s+{E}',
        text
    ):
        role = re.sub(r'\s+', '_', m.group(2).strip()).upper()
        add(m.group(3), role, m.group(1))

    # "X is the son/daughter/nephew/niece/mother/father of Y" -> (X, SON_OF etc, Y)
    for m in re.finditer(
        rf'{E}\s+is\s+(?:the\s+)(son|daughter|nephew|niece|mother|father|brother|sister|cousin|uncle|aunt)\s+of\s+{E}',
        text, re.IGNORECASE
    ):
        rel = m.group(2).upper() + '_OF'
        add(m.group(1), rel, m.group(3))

    # "X became the companion/ally/advisor of Y"
    for m in re.finditer(
        rf'{E}\s+became\s+(?:the\s+)(companion|ally|advisor|consort|apprentice|champion|heir)\s+of\s+{E}',
        text, re.IGNORECASE
    ):
        rel = m.group(2).upper() + '_OF'
        add(m.group(1), rel, m.group(3))

    # "X is a member of Y"
    for m in re.finditer(rf'{E}\s+is\s+a\s+member\s+of\s+{E}', text):
        add(m.group(1), 'MEMBER_OF', m.group(2))

    # "X joined Y"
    for m in re.finditer(rf'{E}\s+joined\s+{E}', text):
        add(m.group(1), 'JOINED', m.group(2))

    # "X betrayed Y"
    for m in re.finditer(rf'{E}\s+betrayed\s+{E}', text):
        add(m.group(1), 'BETRAYED', m.group(2))

    # "X controls Y" / "X governs Y"
    for m in re.finditer(rf'{E}\s+(?:controls|governs|commands|rules)\s+{E}', text):
        add(m.group(1), 'CONTROLS', m.group(2))

    # "X conspired with Y"
    for m in re.finditer(rf'{E}\s+conspired\s+with\s+{E}', text):
        add(m.group(1), 'CONSPIRED_WITH', m.group(2))

    # "X produce/produces Y"
    for m in re.finditer(rf'{E}\s+produces?\s+(?:the\s+)?{E}', text):
        add(m.group(1), 'PRODUCES', m.group(2))

    # "X is the only source of Y"
    for m in re.finditer(rf'{E}\s+is\s+the\s+only\s+source\s+of\s+{E}', text):
        add(m.group(1), 'SOURCE_OF', m.group(2))

    # "X is set on Y" (fictional setting)
    for m in re.finditer(rf'set\s+on\s+(?:the\s+)?{E}', text):
        add('Story', 'SET_ON', m.group(1))

    # "X written/authored by Y"
    for m in re.finditer(rf'{E}\s+(?:was\s+)?written\s+by\s+{E}', text):
        add(m.group(1), 'WRITTEN_BY', m.group(2))

    # "X is the author of Y"
    for m in re.finditer(rf'{E}\s+is\s+the\s+author\s+of\s+{E}', text):
        add(m.group(1), 'AUTHORED', m.group(2))

    # "X became known as Y" (alias)
    for m in re.finditer(rf'{E}\s+became\s+known\s+as\s+{E}', text):
        add(m.group(1), 'KNOWN_AS', m.group(2))

    # "X is also known as Y"
    for m in re.finditer(rf'{E},?\s+also\s+known\s+as\s+{E}', text):
        add(m.group(1), 'ALSO_KNOWN_AS', m.group(2))

    # "X became Y" (title/role change)
    for m in re.finditer(
        rf'{E}\s+became\s+(?:the\s+)?(Emperor|King|Queen|Leader|Champion|Ruler)\s+of\s+{E}',
        text
    ):
        role = m.group(2).upper() + '_OF'
        add(m.group(1), role, m.group(3))

    # "X ride/rides Y"
    for m in re.finditer(rf'{E}\s+ride\s+(?:the\s+)?{E}|{E}\s+rides\s+(?:the\s+)?{E}', text):
        s = m.group(1) or m.group(3)
        o = m.group(2) or m.group(4)
        if s and o:
            add(s, 'RIDES', o)

    # "X is a rival of Y"
    for m in re.finditer(rf'{E}\s+is\s+a\s+rival\s+of\s+{E}', text):
        add(m.group(1), 'RIVAL_OF', m.group(2))

    # "X enables Y" / "X enables interstellar travel" etc.
    for m in re.finditer(rf'{E}\s+enables\s+{E}', text):
        add(m.group(1), 'ENABLES', m.group(2))

    # "X is the most valuable substance" -> skip (too generic)
    # "X is prophesied by Y"
    for m in re.finditer(rf'{E}\s+(?:was\s+)?prophesied\s+by\s+{E}', text):
        add(m.group(1), 'PROPHESIED_BY', m.group(2))


    # "X, heir of Y"
    for m in re.finditer(rf'{E},\s+(?:the\s+)?heir\s+(?:of|to)\s+{E}', text):
        add(m.group(1), 'HEIR_OF', m.group(2))

    # "assigned to govern X" -> (Family/Person, GOVERNS, X)
    for m in re.finditer(rf'{E}.{{1,40}}?assigned\s+to\s+govern\s+{E}', text):
        add(m.group(1), 'GOVERNS', m.group(2))

    # "escapes into X" / "escapes to X"
    for m in re.finditer(rf'{E}.{{1,40}}?escapes?\s+(?:into|to)\s+(?:the\s+)?{E}', text):
        add(m.group(1), 'ESCAPES_TO', m.group(2))

    # "X, in alliance with Y"
    for m in re.finditer(rf'{E}(?:,|.{{1,40}}?)\s+in\s+alliance\s+with\s+(?:the\s+)?{E}', text):
        add(m.group(1), 'ALLIED_WITH', m.group(2))

    # "uniting X"
    for m in re.finditer(rf'{E}.{{1,40}}?uniting\s+(?:the\s+)?{E}', text):
        add(m.group(1), 'UNITES', m.group(2))

    # "reclaim X"
    for m in re.finditer(rf'{E}.{{1,40}}?reclaim\s+{E}', text):
        add(m.group(1), 'RECLAIMS', m.group(2))

    # === GENERIC FALLBACK FOR ARBITRARY TEXT ===
    # If standard rules didn't catch much, do a naive sweep for capitalized entities
    if len(triples) < 5:
        for s in re.split(r'[.!?\n]', text):
            s = s.strip()
            if not s: continue
            cap_phrases = re.findall(E, s)
            # Filter phrases
            cap_phrases = [p for p in cap_phrases if len(p.strip()) > 2 and p.lower() not in JUNK and p.lower() not in LEADING_STRIP]
            # Deduplicate locally while maintaining order
            seen_p = set()
            unique_phrases = []
            for p in cap_phrases:
                if p not in seen_p:
                    unique_phrases.append(p)
                    seen_p.add(p)
                    
            if len(unique_phrases) >= 2:
                for i in range(len(unique_phrases)-1):
                    sub = unique_phrases[i]
                    obj = unique_phrases[i+1]
                    m = re.search(rf'{re.escape(sub)}(.*?){re.escape(obj)}', s)
                    if m:
                        pred_raw = m.group(1).strip()
                        pred = re.sub(r'[^a-zA-Z0-9\s]', '', pred_raw)
                        # Remove articles
                        words = [w for w in pred.split() if w.lower() not in ('a', 'an', 'the', 'some')]
                        pred = " ".join(words).strip()
                        if not pred:
                            pred = "related to"
                        add(sub, pred, obj)

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

def describe_node(entity: str, context_text: str) -> str:
    """Uses LLM to generate a short description of the entity based on the context."""
    prompt = (
        f"Based on the following text, provide a short, 1-2 sentence description of the entity '{entity}'. "
        "Do not include formatting, just the text. If the entity is not mentioned or you cannot determine it, "
        "say 'No detailed description available.'\n\n"
        f"TEXT:\n{context_text}"
    )
    
    if USE_LOCAL_GGUF:
        out = call_local_gguf(prompt)
        if out: return out.strip()
        
    if USE_LOCAL_LLM:
        out = call_local_llm(prompt)
        if out: return out.strip()
        
    if HF_API and HF_MODEL:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=HF_API)
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=HF_MODEL,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Describe node HF call failed: {e}")
            pass
            
    return "Description could not be generated. Please check your LLM configuration."

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

