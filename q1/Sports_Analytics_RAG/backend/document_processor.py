"""
🧠 Document Processing Functions for Compression and Reranking

This file includes:
1. Contextual Compression - removes irrelevant content based on query similarity
2. Reranking - reorders retrieved documents based on semantic relevance
"""

from sklearn.metrics.pairwise import cosine_similarity  # Used to calculate similarity between vectors


# === Function 1: Contextual Compression ===
def compress_document_context(docs, query: str, embeddings, similarity_threshold=0.3):
    """
    Filters out irrelevant sentences from documents by comparing each sentence
    to the query using cosine similarity.

    Args:
        docs (List[Document]): List of LangChain Document objects
        query (str): User query
        embeddings: Embedding model (e.g., HuggingFaceEmbeddings)
        similarity_threshold (float): Cut-off value below which content is ignored

    Returns:
        List of compressed (filtered) documents
    """
    print(f"🗜️ Compressing context (threshold={similarity_threshold})")

    if not docs:
        return docs

    # Convert the full query to an embedding vector
    query_embedding = embeddings.embed_query(query)

    compressed_docs = []

    for doc in docs:
        # Split document into sentences
        sentences = doc.page_content.split('. ')
        relevant_sentences = []

        for sentence in sentences:
            # Skip very short sentences that aren't meaningful
            if len(sentence.strip()) < 20:
                continue

            # Convert sentence to embedding
            sentence_embedding = embeddings.embed_query(sentence)

            # Measure how similar the sentence is to the query
            similarity = cosine_similarity(
                [query_embedding],
                [sentence_embedding]
            )[0][0]

            # Keep only sentences above the similarity threshold
            if similarity > similarity_threshold:
                relevant_sentences.append(sentence)

        # If relevant sentences found, reconstruct the document
        if relevant_sentences:
            compressed_content = '. '.join(relevant_sentences)
            doc.page_content = compressed_content
            compressed_docs.append(doc)

    print(f"  📊 Compressed to {len(compressed_docs)} relevant documents")
    return compressed_docs


# === Function 2: Rerank Documents by Relevance ===
def rerank_documents_by_similarity(docs, query: str, embeddings):
    """
    Reranks the given documents by comparing their content to the query.
    More relevant documents (higher cosine similarity) come first.

    Args:
        docs (List[Document]): List of LangChain documents
        query (str): Original user query
        embeddings: Embedding model to convert text to vectors

    Returns:
        List[Document]: Documents sorted by relevance (highest first)
    """
    print(f"📊 Reranking {len(docs)} documents by similarity")

    if not docs:
        return docs

    # Embed the user's question
    query_embedding = embeddings.embed_query(query)

    scored_docs = []

    for doc in docs:
        # Embed the document’s content
        doc_embedding = embeddings.embed_query(doc.page_content)

        # Compute similarity between query and doc
        similarity = cosine_similarity(
            [query_embedding],
            [doc_embedding]
        )[0][0]

        # Store doc along with its similarity score
        scored_docs.append((doc, similarity))

    # Sort documents by similarity score (highest first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Print top 3 scores (for insight)
    print(f"  🏆 Top similarity scores: {[f'{score:.3f}' for _, score in scored_docs[:3]]}")

    # Return only the documents, now ordered by relevance
    return [doc for doc, score in scored_docs]
