from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from typing import Literal

# Import processing and generation logic
from processing import process_file
from rag_engine import hybrid_retrieve
from quiz_generator import generate_quiz_from_chunks


# Create FastAPI instance
app = FastAPI()

# Allow Streamlit frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev. Restrict in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded documents
UPLOAD_DIR = "data/uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Quiz Generator Backend is running."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Save the uploaded file to the local directory and trigger processing.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the file for embeddings + indexing
    process_file(file_path)

    return {"message": f"File '{file.filename}' uploaded and processed successfully."}


@app.post("/generate")
async def generate_questions(
    topic: str = Form(...),
    q_type: Literal["quiz", "assignment", "test"] = Form(...),
    difficulty: Literal["easy", "medium", "hard"] = Form(...),
    num_questions: int = Form(...),
):
    """
    Hybrid RAG + LLM-based question generation
    """
    try:
        # Step 1: Retrieve relevant content using Hybrid RAG
        chunks = hybrid_retrieve(query=topic, final_k=5)

        if not chunks:
            return JSONResponse(status_code=404, content={"error": "No relevant content found."})

        # Step 2: Generate questions based on type and difficulty
        result = generate_quiz_from_chunks(
            chunks=chunks,
            q_type=q_type,
            difficulty=difficulty,
            count=num_questions
        )

        return {"generated_content": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
