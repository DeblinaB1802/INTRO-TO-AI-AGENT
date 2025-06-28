from call_llm import call_openai

def classify_query(query: str) -> str:
    """Classify user query into these categories (tavily, wikipedia, math, rag)"""
    prompt = f"""You are a smart assistant that classifies user queries into one of the following tools:
    ["tavily", "wikipedia", "math", "rag"]

    - tavily: Use when the query asks for latest updates, news, or anything that needs real-time search.
    - wikipedia: Use when the query asks for general knowledge, definitions, or well-known facts.
    - math: Use when the query involves any mathematical expression, formula, or computation.
    - rag: Use when the query is abstract, explanatory, or not clearly fitting into the above categories.

    Examples:

    Query: What is the capital of Germany?
    Answer: wikipedia

    Query: What is the square root of 144?
    Answer: math
 
    Query: Search for the latest cricket scores
    Answer: tavily

    Query: What is quantum computing?
    Answer: wikipedia

    Query: Solve 2x + 5 = 13
    Answer: math

    Query: Who is Nikola Tesla?
    Answer: wikipedia

    Query: What are the effects of climate change on biodiversity?
    Answer: rag

    Query: Give me current news about AI regulations
    Answer: tavily

    Just respond with category type.
    Query: {query}
    Answer:"""

    messages=[
            {"role": "system", "content": "You are a helpful tool classifier."},
            {"role": "user", "content": prompt}
        ]
    cat = call_openai(messages, model="gpt-4.1-mini")
    return cat

def structure_query(query: str, context: list[dict]) -> str:
    """Re-structuring user query for better understanding of user intent by LLM"""
    structure_query_prompt = f"""
    You are a query optimization assistant.
    Given a user's original query and the surrounding conversation context, rewrite the query to be more specific, 
    unambiguous, and contextually enriched. The restructured query should preserve the user's intent while incorporating 
    any relevant details from the context to improve clarity and precision.
    Use the chat history only when the current user question is a direct follow-up, clarification, or elaboration request 
    (e.g., questions that begin with phrases like 'explain more', 'elaborate', 'what about this', 'why is that', etc.) or 
    has relevance with the context provided.
    Ignore the context entirely when the user introduces a new topic or question unrelated to previous conversation.
    Given:
    - Query: "{query}"
    - Context: "{context}"

    Return a rewritten version of the query that incorporates relevant context, is clear, and specific.
    """
    messages = [{"role": "user", "content": structure_query_prompt}]
    structured_query = call_openai(messages)
    return structured_query