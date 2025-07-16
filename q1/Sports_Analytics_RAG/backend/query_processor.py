"""
🤖 Query Processing and Decomposition Functions

This file handles:
1. Breaking down complex questions into smaller parts
2. Passing each sub-question through the full RAG pipeline (retrieve → compress → rerank → generate answer)
"""

from langchain.prompts import ChatPromptTemplate  # Used to format the prompt for the LLM


# === Function 1: Decompose Complex Queries ===
def decompose_complex_query(query: str, llm):
    """
    Use an LLM to break down a long, multi-part question into simpler sub-questions.

    Args:
        query (str): A complex question (e.g., "Which team has the best defense and how does their goalkeeper compare?")
        llm: The LLM instance used to generate sub-questions (e.g., ChatOpenAI)

    Returns:
        List[str]: List of atomic sub-questions derived from the complex one
    """
    print(f"🔍 Decomposing query: {query}")

    # Prompt template: What we want the LLM to do
    prompt_text = """
    You are a sports analytics expert. Break down the following complex query into simple, atomic sub-questions.
    Each sub-question should focus on one specific aspect and be answerable independently.

    Complex Query: {query}

    Sub-questions (one per line, numbered):
    """

    # Convert the prompt into a LangChain-compatible format
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Chain = prompt → LLM → response
    chain = prompt | llm
    result = chain.invoke({"query": query})  # Ask the LLM to generate sub-questions

    # === Parse the result into a list of sub-questions ===
    sub_questions = []

    for line in result.content.strip().split('\n'):
        # Check if the line starts with a number and contains a sub-question
        if line.strip() and any(char.isdigit() for char in line[:3]):
            clean_question = line.split('.', 1)[-1].strip()
            if clean_question:
                sub_questions.append(clean_question)

    # If no sub-questions were found, just use the original query as fallback
    if not sub_questions:
        sub_questions = [query]

    print(f"  📝 Generated {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"    {i}. {sq}")

    return sub_questions


# === Function 2: Process Each Sub-question Through RAG ===
def process_single_subquestion(
    subquestion: str,
    vectordb,
    embeddings,
    llm,
    similarity_threshold=0.3,
    top_k=5
):
    """
    Full pipeline for answering a single sub-question:
    1. Retrieve documents
    2. Compress irrelevant content
    3. Rerank based on similarity
    4. Generate answer with citations

    Args:
        subquestion (str): A single focused query
        vectordb: Vector database (Chroma, Pinecone, etc.)
        embeddings: Embedding model to convert text into vectors
        llm: Language model used to generate the final answer
        similarity_threshold (float): Cutoff for context compression
        top_k (int): Number of top documents to use for answer generation

    Returns:
        Dict with sub-question, answer, and supporting citations
    """
    print(f"\n🔎 Processing sub-question: {subquestion}")

    # 🔁 Lazy import to avoid circular dependency
    from vector_db import retrieve_relevant_documents
    from document_processor import compress_document_context, rerank_documents_by_similarity
    from response_generator import generate_answer_with_citations

    # Step 1: Retrieve documents most similar to the sub-question
    docs = retrieve_relevant_documents(vectordb, subquestion)

    # Step 2: Apply contextual compression to filter out unrelated sentences
    compressed_docs = compress_document_context(docs, subquestion, embeddings, similarity_threshold)

    # Step 3: Rerank the compressed documents by how relevant they are
    reranked_docs = rerank_documents_by_similarity(compressed_docs, subquestion, embeddings)

    # Step 4: Generate a well-formed answer using the LLM, with citations
    result = generate_answer_with_citations(subquestion, reranked_docs[:top_k], llm)

    # Return the result in a clean dictionary format
    return {
        "sub_question": subquestion,
        "answer": result["answer"],
        "citations": result["citations"]
    }
