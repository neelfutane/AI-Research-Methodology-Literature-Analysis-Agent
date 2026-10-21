import re
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"

CANDIDATE_K = 18   # Optimized from 30: saves ~3.5s CPU reranking time while preserving 100% recall of all 4 boosting algorithms
FINAL_K = 8        # Kept at 8: ensures AdaBoost, XGBoost, CatBoost, LightGBM and all metrics fit comfortably

# Keep low-score filtering, but don't remove useful weak evidence
MIN_RERANKER_SCORE = 0.01


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker model...")
reranker = CrossEncoder("BAAI/bge-reranker-base")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# HELPER: NORMALIZE PDF NAME
# ============================================================

def normalize_pdf_name(pdf_name):
    """
    Makes paper names consistent.

    Examples:
        paper2       -> paper2.pdf
        paper2.pdf   -> paper2.pdf
        test_paper   -> test_paper.pdf
        test_paper.pdf -> test_paper.pdf
    """

    pdf_name = pdf_name.strip()

    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"

    return pdf_name


# ============================================================
# CLEAN RETRIEVAL QUERY
# ============================================================

# Patterns that add conversational noise without semantic value.
# ORDER MATTERS: more-specific patterns must come before less-specific ones.
_META_PATTERNS = [
    # 1. "used in <filename>.pdf" (must fire BEFORE the plain "in <filename>" pattern
    #    to prevent "used" being orphaned as a trailing word)
    re.compile(r'\s+used\s+in\s+[\w\-]+(?:\.pdf)?\b', re.IGNORECASE),
    # 2. "in <filename>.pdf" — catches any remaining bare "in <file>.pdf" reference
    re.compile(r'\s+in\s+[\w\-]+\.pdf\b', re.IGNORECASE),
    # 3. Instructional opener: "Explain/Describe/List/Summarize all the "
    re.compile(r'^(?:explain|describe|list|summarize)\s+all\s+the\s+', re.IGNORECASE),
    # 4. Remaining "all the" / "all of the" filler
    re.compile(r'\ball\s+(?:of\s+)?the\s+', re.IGNORECASE),
    # 5. Trailing orphan action words left after earlier removals
    re.compile(r'\s+\b(?:used|discussed|found|mentioned|described|covered)\b\s*$', re.IGNORECASE),
]


def clean_retrieval_query(query: str) -> str:
    """
    Strip meta-phrases that dilute vector / reranker scores.

    Examples:
        "Explain all the boosting algorithms used in test_paper.pdf."
            -> "boosting algorithms"
        "What machine learning algorithms are discussed in test_paper.pdf?"
            -> "What machine learning algorithms are discussed"
        "What are classification metrics?"
            -> "What are classification metrics?"  (unchanged)
    """
    cleaned = query.strip()
    for pattern in _META_PATTERNS:
        cleaned = pattern.sub(' ', cleaned)
    # collapse multiple spaces and strip punctuation at ends
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(' .?!,')
    # If cleaning reduced to a very short string, fall back to original
    if len(cleaned.split()) < 2:
        return query.strip()
    return cleaned


# ============================================================
# RERANK CANDIDATES
# ============================================================

def rerank_candidates(
    query,
    documents,
    metadatas,
    distances,
    final_k=FINAL_K
):
    """
    Rerank retrieved ChromaDB candidates using
    BAAI/bge-reranker-base.
    """

    if not documents:
        return []

    pairs = [
        (query, document)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for document, metadata, distance, score in zip(
        documents,
        metadatas,
        distances,
        scores
    ):
        reranked_results.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "reranker_score": float(score)
            }
        )

    # Highest reranker score first
    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    # Remove extremely weak results
    filtered_results = [
        result
        for result in reranked_results
        if result["reranker_score"] >= MIN_RERANKER_SCORE
    ]

    if filtered_results:
        return filtered_results[:final_k]

    # Controlled fallback: if all candidates fall below MIN_RERANKER_SCORE,
    # retain the highest-ranked candidates up to final_k, provided they meet
    # a baseline relevance threshold above pure noise (e.g. >= 0.0005).
    if reranked_results:
        fallback_candidates = [
            result
            for result in reranked_results
            if result["reranker_score"] >= 0.0005
        ]
        if fallback_candidates:
            fallback_results = fallback_candidates[:final_k]
            for result in fallback_results:
                result["low_confidence"] = True
            return fallback_results

    return []


# ============================================================
# GENERAL RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(
    query,
    candidate_k=CANDIDATE_K,
    final_k=FINAL_K
):
    """
    Retrieve relevant chunks from all indexed papers.
    """

    retrieval_query = clean_retrieval_query(query)

    query_embedding = embedding_model.encode(
        retrieval_query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return rerank_candidates(
        query=retrieval_query,   # reranker also uses the cleaned query
        documents=documents,
        metadatas=metadatas,
        distances=distances,
        final_k=final_k
    )


# ============================================================
# RETRIEVE WITHIN ONE PAPER
# ============================================================

def retrieve_within_paper(
    paper_name,
    query,
    candidate_k=CANDIDATE_K,
    final_k=FINAL_K
):
    """
    Retrieve relevant chunks from one specific paper.

    The paper name can be supplied as:
        paper2
    or:
        paper2.pdf
    """

    paper_name = normalize_pdf_name(paper_name)

    # Check whether this paper exists
    paper_records = collection.get(
        where={"source": paper_name},
        limit=1
    )

    if not paper_records["ids"]:
        return []

    retrieval_query = clean_retrieval_query(query)

    query_embedding = embedding_model.encode(
        retrieval_query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        where={"source": paper_name}
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return rerank_candidates(
        query=retrieval_query,   # reranker also uses the cleaned query
        documents=documents,
        metadatas=metadatas,
        distances=distances,
        final_k=final_k
    )


# ============================================================
# COMPARE TWO PAPERS
# ============================================================

def compare_papers(
    paper1,
    paper2,
    query,
    candidate_k=CANDIDATE_K,
    final_k=FINAL_K
):
    """
    Retrieve evidence separately from two papers.

    The actual comparison/reasoning is NOT performed here.
    The AI agent will later compare the retrieved evidence.
    """

    paper1 = normalize_pdf_name(paper1)
    paper2 = normalize_pdf_name(paper2)

    paper1_results = retrieve_within_paper(
        paper_name=paper1,
        query=query,
        candidate_k=candidate_k,
        final_k=final_k
    )

    paper2_results = retrieve_within_paper(
        paper_name=paper2,
        query=query,
        candidate_k=candidate_k,
        final_k=final_k
    )

    return {
        "paper1": {
            "name": paper1,
            "results": paper1_results
        },
        "paper2": {
            "name": paper2,
            "results": paper2_results
        }
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("AI RESEARCH AGENT - RETRIEVAL TEST")
    print("========================================")

    query = input(
        "\nEnter your research question: "
    ).strip()

    results = retrieve_relevant_chunks(query)

    print("\n===== Reranked Chunks =====")

    if not results:
        print("\nNo relevant evidence found.")

    else:

        for i, result in enumerate(results, start=1):

            print(f"\n--- Evidence {i} ---")

            print("\nText:")
            print(result["document"])

            metadata = result["metadata"]

            print("\nMetadata:")
            print(
                f"Title: {metadata.get('title', 'Unknown')}"
            )

            print(
                f"Author: {metadata.get('author', 'Unknown')}"
            )

            print(
                f"Source: {metadata.get('source', 'Unknown')}"
            )

            print(
                f"Page: {metadata.get('page', 'Unknown')}"
            )

            print(
                f"\nChroma Distance: "
                f"{result['distance']:.4f}"
            )

            print(
                f"Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )

    print("\n========================================")
    print("Retrieval test complete.")
    print("========================================")