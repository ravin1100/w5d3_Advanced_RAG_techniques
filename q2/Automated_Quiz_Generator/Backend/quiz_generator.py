from typing import List, Literal
from langchain_openai import OpenAI  # You can swap with any other LLM like HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from .env file

# You can set OPENAI_API_KEY in environment or .env
# Alternatively, replace with HuggingFaceHub or local LLM using LangChain wrappers
llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-instruct", openai_api_key=os.getenv("OPENAI_API_KEY"))  # Use gpt-4 if available

# Prompt template with placeholders
prompt_template = PromptTemplate(
    input_variables=["context", "q_type", "difficulty", "count"],
    template="""
You are an expert educational content generator.

Based on the content below:
---------------------
{context}
---------------------

Generate {count} {q_type} questions of {difficulty} difficulty.

Rules:
- For 'quiz': generate MCQs or fill-in-the-blanks.
- For 'assignment': generate subjective, open-ended questions.
- For 'test': generate a mix of question types.
- Include the answer and a short explanation for each question.

Output format:
1. Question
Answer:
Explanation:

Now start generating:
"""
)

def generate_quiz_from_chunks(
    chunks: List[str],
    q_type: Literal["quiz", "assignment", "test"],
    difficulty: Literal["easy", "medium", "hard"],
    count: int
) -> str:
    """
    Given a list of context chunks, generate a quiz/assignment/test.
    Returns a formatted string response from the LLM.
    """
    # Combine chunks into a single context
    full_context = "\n".join(chunks)

    # Build the chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain with inputs
    response = chain.run({
        "context": full_context,
        "q_type": q_type,
        "difficulty": difficulty,
        "count": count
    })

    return response
