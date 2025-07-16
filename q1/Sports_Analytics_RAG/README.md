# Sports Analytics RAG System

A Retrieval Augmented Generation (RAG) system for sports analytics that can answer complex queries about player performance, team statistics, and game insights.

## Features

- Query Decomposition for complex multi-part questions
- Contextual Compression to reduce irrelevant information
- Basic Reranking using semantic similarity scores
- Citation-based responses showing sources
- Interactive UI with visualization of the RAG process
- FastAPI backend with Streamlit frontend

## Project Structure

```
Sports_Analytics_RAG/
├── backend/
│   ├── data/                 # Sports documents
│   ├── chroma_db/           # Vector database
│   ├── data_loader.py       # Document loading functions
│   ├── document_processor.py # Compression and reranking
│   ├── query_processor.py   # Query decomposition
│   ├── vector_db.py         # Vector database operations
│   ├── main.py             # FastAPI application
│   └── requirements.txt     # Project dependencies
└── frontend/
    └── app.py              # Streamlit UI application
```

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the backend directory with:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend application:
```bash
cd frontend
streamlit run app.py
```

3. Open your browser and navigate to:
- Frontend UI: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## Sample Queries

- "What are the top 3 teams in defense and their key defensive statistics?"
- "Compare Messi's goal-scoring rate in the last season vs previous seasons"
- "Which goalkeeper has the best save percentage in high-pressure situations?"

## Implementation Details

### Query Decomposition
- Uses LLM to break down complex queries into simpler sub-questions
- Each sub-question is processed independently

### Contextual Compression
- Filters out irrelevant content based on similarity to query
- Reduces noise in retrieved documents

### Reranking
- Uses semantic similarity to prioritize most relevant documents
- Improves answer accuracy

### Citations
- Every answer includes source documents
- Maintains traceability of information

## API Endpoints

- `POST /process_query`: Process a sports analytics query
- `GET /health`: Health check endpoint

## Frontend Features

- Interactive query input
- Sample query suggestions
- Visualization of RAG processing steps
- Tabbed results view
- Citation display
- Processing metrics visualization