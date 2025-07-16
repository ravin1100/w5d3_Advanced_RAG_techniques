from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from processing import bm25_index, bm25_corpus, embedding_model_name
import numpy as np
from typing import List, Dict

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

# (Optional) Load cross-encoder model for reranking (heavy model, so only use if needed)
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Set up Chroma vector DB (same location as used in processing.py)
CHROMA_DB_DIR = "data/chroma_db"

vectorstore = Chroma(
    collection_name="quiz_docs",
    embedding_function=embedding_function,
    persist_directory=CHROMA_DB_DIR
)

def retrieve_dense(query: str, k: int = 5) -> List[Dict]:
    """
    Uses ChromaDB to retrieve top-k semantically similar chunks.
    """
    results = vectorstore.similarity_search(query, k=k)
    return [{"text": r.page_content, "score": 1.0, "source": "dense"} for r in results]

def retrieve_sparse(query: str, k: int = 5) -> List[Dict]:
    """
    Uses BM25 to retrieve top-k keyword-relevant chunks.
    """
    if not bm25_index:
        print("⚠️ BM25 index is empty. Make sure a document is uploaded and processed.")
        return []

    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    # Get top-k highest scoring docs
    top_k_indices = np.argsort(scores)[-k:][::-1]

    return [
        {"text": bm25_corpus[i], "score": scores[i], "source": "sparse"}
        for i in top_k_indices
    ]

def rerank_with_cross_encoder(query: str, docs: List[Dict]) -> List[Dict]:
    """
    Optional: Reranks combined results using a cross-encoder model.
    This can improve the ranking order, but is slower.
    """
    # Uncomment below lines to use cross-encoder if needed
    # pairs = [(query, doc["text"]) for doc in docs]
    # scores = cross_encoder.predict(pairs)

    # for i in range(len(docs)):
    #     docs[i]["score"] = float(scores[i])

    # Sort by score descending
    return sorted(docs, key=lambda x: x["score"], reverse=True)

def hybrid_retrieve(query: str, k_dense: int = 5, k_sparse: int = 5, final_k: int = 5) -> List[str]:
    """
    Performs hybrid retrieval using both dense and sparse methods,
    merges results, and returns top `final_k` chunks.
    """
    dense_results = retrieve_dense(query, k_dense)
    sparse_results = retrieve_sparse(query, k_sparse)

    # Merge results and remove duplicates (based on text content)
    all_results = {r["text"]: r for r in dense_results + sparse_results}
    merged_results = list(all_results.values())

    # Optional reranking
    reranked = rerank_with_cross_encoder(query, merged_results)

    # Return top `final_k` text chunks
    return [r["text"] for r in reranked[:final_k]]
