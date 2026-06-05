"""Module 8 — Tuesday Stretch (Honors Track): Query Router.

Build a routing layer that classifies an incoming query into one of three
types and dispatches it to a different retrieval pipeline.
"""

from __future__ import annotations
import re
import weaviate
from sentence_transformers import SentenceTransformer

# Provided helper functions from the stretch repo
from retrieval_helpers import bm25_search, dense_search, hybrid_search

# Global embedder used by the autograder and local testing
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def classify_query(query: str) -> str:
    """Return one of "factoid", "semantic", "mixed".

    Implementation: Rule-based heuristic.
    - Factoid: High presence of capitalized entities, digits (versions/codes), 
      or very short keyword-dense strings.
    - Semantic: Natural language markers (how, why, what), high word count, 
      and lack of specific identifiers.
    - Mixed: Queries that don't hit strong thresholds for either.
    """
    query_clean = query.strip()
    words = query_clean.split()
    word_count = len(words)
    
    # Heuristics for Semantic (Paraphrastic) intent
    semantic_indicators = ["how", "why", "what", "best", "way", "explain", "difference"]
    has_semantic_word = any(w.lower() in semantic_indicators for w in words)
    
    # Heuristics for Factoid intent
    # 1. Regex for capitalized multi-word phrases (Entities like "Android Activity")
    has_entities = bool(re.search(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', query_clean))
    # 2. Presence of digits (Error codes, versions like "v3.0", "404")
    has_digits = bool(re.search(r'\d', query_clean))
    
    # Dispatch Logic
    if has_semantic_word or word_count > 8:
        # Long queries or "how-to" questions benefit from dense vector space
        return "semantic"
    
    if (has_entities or has_digits or word_count <= 3) and not has_semantic_word:
        # Specific identifiers or short keyword queries benefit from BM25 lexical match
        return "factoid"
    
    # Default to hybrid for balanced queries
    return "mixed"


def routed_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dispatch to BM25 / dense / hybrid based on classify_query(query).

    Return the ordered list of doc_id strings, length <= k.
    """
    kind = classify_query(query)
    
    if kind == "factoid":
        # Dispatched to BM25 for exact keyword/identifier matching
        return bm25_search(client, query, k)
    
    elif kind == "semantic":
        # Dispatched to Dense for semantic/conceptual matching
        return dense_search(client, query, k, embedder)
    
    else:
        # Dispatched to Hybrid (alpha=0.5) for mixed intent
        return hybrid_search(client, query, k, embedder, alpha=0.5)

if __name__ == "__main__":
    # Example local testing block
    client = weaviate.Client("http://localhost:8080")
    test_queries = [
        "How do I refactor a long method in Java?", # Semantic
        "Android Activity onPause v2.1",            # Factoid
        "Git merge conflict resolution"             # Mixed
    ]
    
    for q in test_queries:
        print(f"Query: {q} | Route: {classify_query(q)}")