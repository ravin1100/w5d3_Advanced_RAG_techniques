import os
import glob
from typing import List
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from docx import Document
from pypdf import PdfReader

# Global paths
UPLOAD_DIR = "data/uploaded_docs"
CHROMA_DB_DIR = "data/chroma_db"

# Initialize embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Global BM25 store
bm25_corpus = []
bm25_tokenized = []
bm25_index = None

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text(file_path: str) -> str:
    """
    Reads text from PDF, DOCX, or TXT based on file extension.
    """
    if file_path.endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.endswith(".docx"):
        return read_docx(file_path)
    elif file_path.endswith(".txt"):
        return read_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Breaks text into smaller chunks using LangChain's text splitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def store_in_chroma(chunks: List[str], doc_id: str):
    """
    Store embedded chunks in Chroma vector database.
    """
    vectorstore = Chroma(
        collection_name="quiz_docs",
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_DIR
    )

    metadatas = [{"doc_id": doc_id, "chunk_id": i} for i in range(len(chunks))]

    vectorstore.add_texts(texts=chunks, metadatas=metadatas)
    # vectorstore.persist()

def index_with_bm25(chunks: List[str]):
    """
    Index chunks using BM25 for sparse keyword-based search.
    """
    global bm25_corpus, bm25_tokenized, bm25_index
    bm25_corpus.extend(chunks)
    bm25_tokenized = [doc.lower().split() for doc in bm25_corpus]
    bm25_index = BM25Okapi(bm25_tokenized)

def process_file(file_path: str):
    """
    Main entry to extract, chunk, embed, and store a document.
    """
    print(f"Processing: {file_path}")

    # Step 1: Extract raw text
    raw_text = extract_text(file_path)

    # Step 2: Chunk text
    chunks = chunk_text(raw_text)

    # Step 3: Generate a document ID
    doc_id = str(uuid4())

    # Step 4: Store in Chroma (dense retrieval)
    store_in_chroma(chunks, doc_id)

    # Step 5: Index with BM25 (sparse retrieval)
    index_with_bm25(chunks)

    print(f"✅ Document processed and stored. ID: {doc_id}")

    return doc_id, len(chunks)
