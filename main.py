

# make sure to add the cache and config file in the same folder before running the code
#modify neo4j uri accordingly for auradb "neo4j+ssc://...."


import json
import asyncio
import os
from typing import List
import aiohttp
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import AsyncGraphDatabase
import config

# Configurable Parameters
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME
CACHE_FILE = "embeddings_cache.json"#add this file in the same folder 

# Initialize clients
client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Ensure Pinecone index exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# Embedding Cache
_embedding_cache = {}
_cache_lock = asyncio.Lock()

def _load_cache_from_file():
    global _embedding_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _embedding_cache = json.load(f)
                print(f"Loaded {_embedding_cache and len(_embedding_cache)} cached embeddings.")
        except Exception as e:
            print(f"Warning: failed to load cache: {e}")

def _save_cache_to_file():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_embedding_cache, f)
    except Exception as e:
        print(f"Warning: failed to save cache: {e}")

_load_cache_from_file()

# Async helper functions
async def embed_text_async(text: str) -> List[float]:
    """Get embedding asynchronously, with caching."""
    async with _cache_lock:
        if text in _embedding_cache:
            return _embedding_cache[text]

    async with aiohttp.ClientSession() as session:
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": EMBED_MODEL, "input": [text]}
        async with session.post(url, headers=headers, json=payload) as resp:
            data = await resp.json()
            vec = data["data"][0]["embedding"]

    async with _cache_lock:
        _embedding_cache[text] = vec
        _save_cache_to_file()

    return vec

async def pinecone_query_async(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding (async)."""
    vec = await embed_text_async(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Pinecone top {top_k} results:", len(res["matches"]))
    return res["matches"]

# Improved Neo4j Retrieval
async def fetch_graph_context_async(node_ids: List[str], neighborhood_depth=1):
    """Fetch direct neighbors (depth 1) for all given nodes in one query."""
    if not node_ids:
        return []

    facts = []
    async with driver.session() as session:
        query = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.id IN $node_ids
        RETURN DISTINCT
            n.id AS source,
            type(r) AS rel,
            m.id AS target_id,
            m.name AS target_name,
            m.description AS target_desc,
            labels(m) AS labels
        LIMIT 50
        """
        recs = await session.run(query, node_ids=node_ids)
        async for r in recs:
            facts.append({
                "source": r["source"],
                "rel": r["rel"],
                "target_id": r["target_id"],
                "target_name": r["target_name"],
                "target_desc": (r["target_desc"] or "")[:400],
                "labels": r["labels"]
            })
    print("DEBUG: Graph facts:", len(facts))
    return facts

# Search Summary Helper
async def summarize_search_results_async(pinecone_matches, top_k=5):
    """Generate a short natural-language summary of the top-K semantic matches."""
    if not pinecone_matches:
        return "No relevant search results found."

    # Take only the top-K matches
    top_matches = pinecone_matches[:top_k]

    descriptions = []
    for m in top_matches:
        meta = m.get("metadata", {})
        line = f"{meta.get('name','Unnamed')} ({meta.get('type','Unknown')}) in {meta.get('city','Unknown')}"
        descriptions.append(line)

    summary_prompt = [
        {
            "role": "system",
            "content": "You are summarizing search results for a travel assistant. "
                       "Describe them briefly in one or two sentences and do not cite node ids."
        },
        {
            "role": "user",
            "content": "Summarize these entities:\n" + "\n".join(descriptions)
        }
    ]

    try:
        resp = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=summary_prompt,
            max_tokens=60,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("Warning: summary generation failed:", e)
        return "Summary unavailable."


# Prompt Structure
def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build structured prompt that groups semantic and graph context."""
    system = (
        "You are a smart and modern travel assistant. "
        "Use both the semantic search results and graph relationships "
        "to provide a descriptive, accurate, and helpful answer. "
        "Cite tips accordingly"
        
        
    )

    # Group related facts by source id
    grouped_facts = {}
    for f in graph_facts:
        grouped_facts.setdefault(f["source"], []).append(f)

    # Build structured context per matched entity
    entity_contexts = []
    for m in pinecone_matches:
        mid = m["id"]
        meta = m["metadata"]
        related = grouped_facts.get(mid, [])
        relations_text = "\n".join(
            [f"  - {f['rel']} → {f['target_name']} ({f['target_id']})" for f in related]
        ) or "  - No related entities found."
        entry = (
            f"[{mid}] {meta.get('name','(Unnamed)')} ({meta.get('type','')})\n"
            f"Description: {meta.get('description','')[:300]}\n"
            f"Connections:\n{relations_text}"
        )
        entity_contexts.append(entry)

    user_content = (
        f"User query: {user_query}\n\n"
        "Relevant entities and their connections:\n\n" +
        "\n\n".join(entity_contexts) +
        "\n\nPlease provide a brief, contextually grounded answer."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]

# OpenAI Chat
async def call_chat_async(prompt_messages):
    """Async OpenAI chat call."""
    resp = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.2
    )
    return resp.choices[0].message.content

# Interactive chat
async def interactive_chat_async():
    print("Welcome to Hybrid travel assistant")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        pinecone_matches = await pinecone_query_async(query)
        match_ids = [m["id"] for m in pinecone_matches]

        graph_facts = await fetch_graph_context_async(match_ids)

        # Generate and print search summary for top 5 matches
        search_summary_text = await summarize_search_results_async(pinecone_matches, top_k=5)
        print("\n=== Top 5 Semantic Matches Summary ===\n")
        print(search_summary_text)
        print("\n==============================\n")

        prompt = build_prompt(query, pinecone_matches, graph_facts)
        answer = await call_chat_async(prompt)

        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

# Entry point
if __name__ == "__main__":
    asyncio.run(interactive_chat_async())
