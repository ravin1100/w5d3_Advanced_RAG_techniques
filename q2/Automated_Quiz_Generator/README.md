# 🤖 Automated Quiz Generator

An AI-powered system that allows educators to upload course material and generate **quizzes**, **assignments**, and **tests** using **Hybrid Retrieval-Augmented Generation (RAG)** with **LangChain** and **LLMs**.


## 🌟 Features

- Upload educational documents (`.pdf`, `.docx`, `.txt`)
- Hybrid RAG: Combines dense (Sentence Transformers) and sparse (BM25) retrieval
- Dynamic chunking and compression
- LangChain + LLMs for question generation
- Generates MCQs, open-ended questions, and mixed tests
- Difficulty levels: Easy, Medium, Hard
- Clean and interactive UI with Streamlit


## 🚀 How It Works

```mermaid
graph TD
    A["📁 Upload Document (PDF/DOCX/TXT via Streamlit)"] --> B["🔄 Send to FastAPI Backend (/upload)"]
    B --> C["📃 Extract Raw Text from File"]
    C --> D["✂️ Dynamically Chunk Text into Passages"]
    D --> E1["🔗 Generate Dense Embeddings (SentenceTransformer)"]
    D --> E2["🔎 Index Chunks for BM25 (Sparse Tokens)"]

    E1 --> F1["📦 Store Embeddings in ChromaDB"]
    E2 --> F2["📚 Store BM25 Corpus in Memory"]

    subgraph "🧠 Retrieval Pipeline"
        G1["🔍 Dense Retrieval via ChromaDB"]
        G2["📖 Sparse Retrieval via BM25"]
        G1 --> H["🧮 Merge Results & Deduplicate"]
        G2 --> H
        H --> I["📊 (Optional) Rerank with Cross-Encoder"]
    end

    subgraph "💬 LLM Generation via LangChain"
        I --> J["📝 Build Prompt using Retrieved Context"]
        J --> K["🤖 Invoke LLM (e.g., OpenAI, HF, Ollama)"]
        K --> L["🧾 Generate Quiz / Assignment / Test"]
    end

    L --> M["📤 Return Output to Streamlit Frontend"]
    M --> N["📄 Display Questions with Answers & Explanations"]

```


## 🖼️ UI Preview

Here’s what the app looks like in action:

### 📤 Upload Section

Upload educational documents (`.pdf`, `.docx`, `.txt`)

![Upload Section](./UI-Snapshots/upload_section.png)


### 🧠 Generation Form

Select question type, difficulty, number of questions, and topic

![Generation Form](./UI-Snapshots/generation_form.png)


### 📝 Output Viewer

Generated quiz, assignment, or test shown in a clean textbox

![Output Viewer](./UI-Snapshots/output_viewer.png)



## 🧱 Folder Structure

```

Automated_Quiz_Generator/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── processing.py        # File parsing, chunking, and embedding
│   ├── rag_engine.py        # Hybrid RAG logic (BM25 + dense + rerank)
│   ├── quiz_generator.py    # LangChain-based quiz generation
│
├── frontend/
│   └── app.py               # Streamlit user interface
│
├── data/
│   └── uploaded_docs/       # Uploaded files stored here
│
├── vector_db/               # (Optional) ChromaDB persistent storage
│
├── requirements.txt         # Dependencies
└── README.md                

```

# 🧠 Tech Stack

This project combines modern NLP tools, vector databases, and LLM orchestration frameworks to build a hybrid RAG-based quiz/assignment generator.


## ⚙️ Technologies Used

| **Area**           | **Tech**                              |
|--------------------|----------------------------------------|
| 🖥️ Backend         | [FastAPI]() - High-performance web API framework |
| 🎛️ Frontend        | [Streamlit]() - Interactive app builder for ML/data apps |
| 🔡 Embeddings       | [sentence-transformers]() - Dense vector representations |
| 🧮 Sparse Search    | [BM25]() (`rank_bm25`) - Lexical retrieval |
| 🧠 Vector Store     | [ChromaDB]() - Lightweight and persistent vector DB |
| 🔗 LLM Integration  | [LangChain]() + [OpenAI]() - RAG & question generation |
| 📄 PDF Parsing      | [PyPDF2]() / [pdfplumber]() - Text extraction from PDFs |
| 📃 DOCX Support     | [python-docx]() - Extract text from `.docx` files |
| 🚀 Deployment       | [Uvicorn]() - ASGI server to run FastAPI |


## 🚀 How to Run This Application

Follow these steps to get the backend and frontend running locally.

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/ravin1100/w5d3_Advanced_RAG_techniques.git

cd ./q1/Automated-Quiz-Generator

```

### ✅ 2. Create & Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

```

### ✅ 3. Install Requirements

```bash
pip install -r requirements.txt
```

### ✅ 4. Run the FastAPI Backend

```bash
uvicorn backend.main:app --reload
```
This will start the backend server on:
📍 http://localhost:8000

### ✅ 5. Run the Streamlit Frontend
Open a new terminal (while backend is still running):

```bash
streamlit run frontend/app.py
```
The frontend will open at:
🌐 http://localhost:8501

### ✅ 6. Upload a Document and Generate Quiz

- Drag and drop a .pdf, .docx, or .txt file.

- Choose the question type and difficulty.

- Click Generate to get questions powered by LLM + Hybrid RAG.

