"""
🗃️ Vector Database Operations

This module handles:
1. Creating a vector database from document chunks
2. Loading an existing vector DB (from disk)
3. Retrieving relevant documents for a given query
"""

from langchain_community.vectorstores import Chroma  # ChromaDB integration with LangChain


# === Function 1: Create and Persist a Vector Database ===
def create_vector_database(docs, embeddings, persist_dir="./chroma_db"):
    """
    Converts document chunks into vector embeddings and stores them in a persistent ChromaDB database.

    Args:
        docs (List[Document]): Chunked documents to store
        embeddings: Embedding model (e.g., HuggingFaceEmbeddings)
        persist_dir (str): Directory to save the ChromaDB

    Returns:
        Chroma: The created vector store object
    """
    print(f"🗃️ Creating vector database in {persist_dir}")

    # Create a Chroma vector database from the given documents
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Save to disk so we can reload later
    vectordb.persist()

    print("  ✅ Vector database created and persisted")
    return vectordb


# === Function 2: Load an Existing Vector Database ===
def load_existing_database(embeddings, persist_dir="./chroma_db"):
    """
    Loads a previously saved ChromaDB from disk.

    Args:
        embeddings: Embedding function used during creation (must be the same)
        persist_dir (str): Directory where the database is stored

    Returns:
        Chroma: The loaded vector store object
    """
    print(f"📂 Loading existing database from {persist_dir}")

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    print("  ✅ Database loaded")
    return vectordb


# === Function 3: Retrieve Top-K Relevant Documents ===
def retrieve_relevant_documents(vectordb, query: str, k: int = 10):
    """
    Given a query, fetches the top-K most relevant documents using vector similarity search.

    Args:
        vectordb: The Chroma vector store
        query (str): The user question or sub-question
        k (int): Number of top documents to retrieve

    Returns:
        List[Document]: Top-K most relevant document chunks
    """
    print(f"📚 Retrieving documents for: {query[:50]}...")  # Show preview of query

    # Create a retriever with the given number of top results
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # Get the most relevant documents
    docs = retriever.get_relevant_documents(query)

    print(f"  📖 Retrieved {len(docs)} documents")
    return docs
