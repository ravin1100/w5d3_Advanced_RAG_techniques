import streamlit as st
import requests

# Backend URL (change if deployed)
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Quiz Generator", layout="centered")

st.title("📘 AI-Powered Quiz / Assignment / Test Generator")
st.markdown("Upload educational content and generate intelligent assessments using Hybrid RAG + LLMs.")

# Section 1: Upload File
st.header("1️⃣ Upload Educational Document")
uploaded_file = st.file_uploader("Upload a .pdf, .docx, or .txt file", type=["pdf", "docx", "txt"])

if uploaded_file:
    if st.button("📤 Upload and Process"):
        with st.spinner("Uploading and processing document..."):
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            )

            if response.status_code == 200:
                st.success("✅ File uploaded and processed successfully!")
            else:
                st.error("❌ Upload failed.")

st.markdown("---")

# Section 2: Generate Questions
st.header("2️⃣ Generate Quiz / Assignment / Test")

topic = st.text_input("Topic or Learning Objective", placeholder="e.g., Photosynthesis, World War II")
q_type = st.selectbox("Question Type", ["quiz", "assignment", "test"])
difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
num_questions = st.slider("Number of Questions", min_value=1, max_value=20, value=5)

if st.button("🧠 Generate"):
    if not topic:
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating questions..."):
            response = requests.post(
                f"{BACKEND_URL}/generate",
                data={
                    "topic": topic,
                    "q_type": q_type,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                }
            )

            if response.status_code == 200:
                output = response.json()["generated_content"]
                st.success("✅ Generated Successfully!")
                st.markdown("### ✍️ Output:")
                st.text_area("Generated Questions", value=output, height=400)
            else:
                st.error("❌ Generation failed.")
