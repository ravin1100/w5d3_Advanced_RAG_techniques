"""
🛠️ Utility Functions for the Sports Analytics RAG System

These helper functions improve code readability and organize how output is shown to the user.
"""

# === Function 1: Print a Line Separator ===
def print_separator(char="=", length=80):
    """
    Prints a horizontal separator line.
    Useful for formatting console output.

    Args:
        char (str): Character to repeat (default '=')
        length (int): How many times to repeat the character
    """
    print(char * length)


# === Function 2: Display Query Being Processed ===
def print_query_header(query: str):
    """
    Prints a formatted header when processing a new sports query.

    Args:
        query (str): The user query string
    """
    print_separator()
    print(f"🏈 PROCESSING SPORTS QUERY: {query}")
    print_separator()


# === Function 3: Print Final Answer and Citations ===
def print_final_results(final_answer: str, citations: list):
    """
    Prints the final answer and related citations to the console in a clean format.

    Args:
        final_answer (str): The generated response from the model
        citations (list): List of citations like ['[1] team_stats.txt', '[2] messi_goals.txt']
    """
    print(f"\n📋 FINAL ANSWER:")
    print(final_answer)

    print(f"\n📚 CITATIONS:")
    for citation in citations:
        print(f"  {citation}")
