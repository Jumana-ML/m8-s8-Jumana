# Routing Report — Module 8 Tuesday Stretch

## 1. Per-Query-Type Classifier Accuracy

I implemented a **rule-based classifier** that uses a combination of linguistic markers and structural heuristics. 
- **Semantic Heuristics:** The classifier flags queries as "semantic" if they contain natural language question starters (e.g., "how", "why", "best way") or if the query length exceeds 8 words, suggesting a descriptive intent.
- **Factoid Heuristics:** The classifier uses regex to identify capitalized multi-word entities (e.g., "Android Activity") or digits (e.g., "v2.1", "404"), and flags very short queries (3 words or fewer) as factoid-heavy.
- **Mixed:** This serves as the fallback for balanced technical queries.

On a hand-labeled subset of 20 queries from the evaluation set, the classifier achieved **85% accuracy**. It correctly identified 9/10 semantic queries and 8/10 factoid queries. The primary confusion occurred with short, conceptual queries that lacked "how/why" markers, which were sometimes misclassified as factoid due to their brevity.

## 2. Routed Retriever Metrics

Comparison table:

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 (baseline) | 0.567 | 0.650 | 0.549 |
| Dense (baseline) | 0.900 | 0.933 | 0.670 |
| Hybrid α=0.5 (baseline) | 0.850 | 0.983 | 0.698 |
| **Routed** | **0.916** | **0.983** | **0.725** |

The routed retriever outperformed all baselines in MRR and matched the top performance in Recall@10. This confirms that dispatching specific queries to the appropriate engine is more effective than using a "one-size-fits-all" approach.

## 3. When Does Routing Win, When Does It Lose, Why

**Where Routing Wins:**
Routing wins most significantly on highly specific technical identifiers. For example, the query **"Android Activity onPause v2.1"** contains a specific API method and a version number. The router correctly identified this as a `factoid` and dispatched it to **BM25**. While the Dense retriever might find several semantically similar "lifecycle" documents, BM25 provides a perfect Rank 1 match for the exact tokens, boosting the overall MRR.

**Where Routing Loses:**
The primary failure mode is **misclassification**. For a query like **"clean code rules"**, the classifier may route it to BM25 because it is short (3 words) and lacks semantic keywords. However, the "gold document" for this query might be a long post about "refactoring and software craftsmanship" which does not contain the exact phrase "clean code rules." In this case, the **Dense** retriever would have been superior, but the router failed to recognize the semantic nature of the short query.

**Future Improvements:**
To improve the router, I would move away from rigid word-count heuristics and toward a **small-model classifier** (like a fine-tuned DistilBERT). A model-based classifier would better understand that a 3-word query can still be "semantic" in nature, reducing the misclassification of short, conceptual queries. Additionally, implementing a "confidence score" where low-confidence classifications default to **Hybrid** would make the system more robust against edge cases.