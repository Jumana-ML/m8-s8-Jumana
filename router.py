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
    # Added "does" and "is" to better capture natural language questions
    semantic_indicators = ["how", "why", "what", "best", "way", "explain", "difference", "does", "is"]
    has_semantic_word = any(w.lower() in semantic_indicators for w in words)
    
    # Heuristics for Factoid intent
    has_entities = bool(re.search(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', query_clean))
    has_digits = bool(re.search(r'\d', query_clean))
    
    # --- Dispatch Logic ---
    
    # 1. Purely descriptive queries with no technical identifiers go to Semantic (Dense)
    if (has_semantic_word or word_count > 10) and not (has_entities or has_digits):
        return "semantic"
    
    # 2. Very short queries with specific IDs/Entities go to Factoid (BM25)
    # Restricted to word_count <= 4 to prevent complex questions from being misrouted
    if word_count <= 4 and (has_entities or has_digits) and not has_semantic_word:
        return "factoid"
    
    # 3. Everything else (Complex technical questions) goes to Mixed (Hybrid)
    # This is the "Safe Route" that ensures the router matches the best baseline
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