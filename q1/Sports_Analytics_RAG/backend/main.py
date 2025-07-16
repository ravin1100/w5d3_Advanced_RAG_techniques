"""
🏈 Sports Analytics RAG System – Main Execution File
This file runs the full pipeline:
- Loads or sets up the vector database
- Accepts complex sports-related questions
- Breaks them into simpler parts
- Finds answers using relevant documents
- Returns a combined, cited response
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import our RAG components
from query_processor import decompose_complex_query, process_single_subquestion
from data_loader import load_documents_from_folder, chunk_documents
from document_processor import compress_document_context, rerank_documents_by_similarity
from vector_db import init_vector_store

# Initialize FastAPI app
app = FastAPI(title="Sports Analytics RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class SubQuestionResponse(BaseModel):
    sub_question: str
    answer: str
    citations: List[Dict[str, str]]

class QueryResponse(BaseModel):
    original_query: str
    sub_questions: List[SubQuestionResponse]
    processing_steps: Dict[str, str]

# Global variables for RAG components
vector_store = None
llm = None
embeddings = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global vector_store, llm, embeddings
    # Initialize your components here
    # This is placeholder - you'll need to add actual initialization code
    pass

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a complex sports analytics query through the RAG pipeline
    """
    try:
        # 1. Query Decomposition
        sub_questions = decompose_complex_query(request.query, llm)
        
        # 2. Process each sub-question
        results = []
        processing_steps = {}
        
        for sub_q in sub_questions:
            # Process the sub-question and track each step
            result = process_single_subquestion(
                sub_q,
                vector_store,
                embeddings,
                llm
            )
            results.append(result)
            
            # Track processing steps for visualization
            processing_steps[f"sub_question_{len(results)}"] = {
                "query_decomposition": sub_q,
                "retrieved_docs": "Number of docs retrieved: X",  # You'll need to add actual numbers
                "compression": "Compressed to Y relevant documents",
                "reranking": "Top similarity scores: [...]"
            }
        
        return QueryResponse(
            original_query=request.query,
            sub_questions=results,
            processing_steps=processing_steps
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
