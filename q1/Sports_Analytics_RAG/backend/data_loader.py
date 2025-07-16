"""
📂 Document Loading and Chunking Functions
This file contains functions to:
1. Load text files from a folder
2. Split (chunk) long text documents into smaller parts for processing
"""

import os  # Built-in Python module to work with files and folders

# Loader to read plain text files using LangChain
from langchain_community.document_loaders import TextLoader
# ✅ If you're on Windows and facing 'pwd' error, replace above with:
# from langchain_community.document_loaders.text import TextLoader

# Used to split large documents into smaller overlapping chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter


# === Function 1: Load .txt files from a folder ===
def load_documents_from_folder(folder_path: str):
    """
    Loads all .txt documents from the given folder.

    Args:
        folder_path (str): Path to folder containing .txt files

    Returns:
        List of Document objects (LangChain format) with source metadata
    """
    all_docs = []

    print(f"📁 Loading documents from {folder_path}")

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(folder_path, filename)

            # Use LangChain's TextLoader to load content from the file
            loader = TextLoader(file_path)
            docs = loader.load()

            # Add metadata: tag each document with its source filename
            for doc in docs:
                doc.metadata["source"] = filename

            all_docs.extend(docs)
            print(f"  📄 Loaded {filename}")

    return all_docs


# === Function 2: Split long documents into smaller chunks ===
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """
    Splits documents into smaller overlapping chunks to improve retrieval accuracy.

    Args:
        docs (List[Document]): List of documents to be chunked
        chunk_size (int): Number of characters per chunk
        chunk_overlap (int): Number of characters to overlap between chunks

    Returns:
        List of smaller document chunks
    """
    print(f"✂️ Chunking documents (size={chunk_size}, overlap={chunk_overlap})")

    # Initialize the chunker
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []

    # Split each document individually
    for doc in docs:
        chunks = splitter.split_documents([doc])  # Returns a list of smaller docs
        all_chunks.extend(chunks)

    print(f"  📊 Created {len(all_chunks)} chunks")
    return all_chunks
