"""
📝 Response Generation and Citation Functions

This file handles:
1. Creating final answers using retrieved documents
2. Adding citations to show where facts came from
3. Combining sub-answers into one final answer
"""

from langchain.prompts import ChatPromptTemplate  # Used to format prompts for the LLM


# === Function 1: Generate Answer with Citations ===
def generate_answer_with_citations(query: str, docs, llm):
    """
    Generates an answer to a sub-question using relevant documents.
    Adds citations (like [1], [2]) that refer to the source of each supporting document.

    Args:
        query (str): The sub-question to answer
        docs (List[Document]): List of relevant documents to use
        llm: The language model to generate the answer (e.g., ChatOpenAI)

    Returns:
        Dict with the generated answer and its citations
    """
    print(f"💬 Generating answer for: {query[:50]}...")  # Show first 50 chars of the query

    # If no documents found, return a fallback message
    if not docs:
        return {
            "answer": "I couldn't find relevant information to answer your question.",
            "citations": []
        }

    # Prepare the input context for the LLM, including source numbers
    context_parts = []
    citations = []

    for i, doc in enumerate(docs):
        # Get the source file name or fallback to "unknown"
        source = doc.metadata.get("source", "unknown")

        # Format the chunked content with [index] label for citation
        context_parts.append(f"[{i+1}] {doc.page_content}")

        # Track the source file with the same index
        citations.append(f"[{i+1}] {source}")

    # Join all chunks to form a single input string for the prompt
    context = "\n\n".join(context_parts)

    # Prompt template to instruct the LLM on how to respond
    prompt_text = """
    You are a sports analytics expert. Answer the question using the provided context.

    IMPORTANT: 
    - Cite your sources using the [number] format from the context
    - Be specific with statistics and comparisons
    - If information is insufficient, say so
    - Keep response focused and analytical

    Question: {query}

    Context:
    {context}

    Answer with citations:
    """

    # Create and run the prompt
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    result = chain.invoke({"query": query, "context": context})

    print(f"  ✅ Answer generated with {len(citations)} citations")

    return {
        "answer": result.content,
        "citations": citations
    }


# === Function 2: Combine All Sub-Question Results into One Answer ===
def combine_subquestion_results(original_query: str, sub_results, llm):
    """
    Merges all sub-question answers into a single comprehensive answer.

    Args:
        original_query (str): The original user question
        sub_results (List[Dict]): List of sub-question results (each with answer + citations)
        llm: LLM used to synthesize the final response

    Returns:
        Dict with the full final answer and all unique citations
    """
    print(f"🔄 Combining results for final answer")

    combined_context = []
    all_citations = []

    # Prepare formatted input for the final answer prompt
    for i, result in enumerate(sub_results):
        combined_context.append(f"Sub-question {i+1}: {result['sub_question']}")
        combined_context.append(f"Answer: {result['answer']}")

        # Collect all citations from individual sub-answers
        all_citations.extend(result['citations'])

    # Merge the sub-question results into a single string
    context = "\n\n".join(combined_context)

    # Prompt to guide the LLM to synthesize a complete response
    prompt_text = """
    You are a sports analytics expert. Provide a comprehensive answer to the original query 
    by synthesizing the sub-question answers below.

    Original Query: {original_query}

    Sub-question Results:
    {context}

    Provide a comprehensive, well-structured answer that addresses all aspects of the original query:
    """

    # Create and run the final combination prompt
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    result = chain.invoke({"original_query": original_query, "context": context})

    return {
        "answer": result.content,
        "citations": list(set(all_citations))  # Remove duplicate citations
    }
