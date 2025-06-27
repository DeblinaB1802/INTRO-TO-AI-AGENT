from call_llm import call_openai
from sentence_transformers import SentenceTransformer, util

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def plan_execute_refine_math(query, context):
    """
    Use the Plan-Execute-Refine pattern to solve mathematical problems.
    Ensures step-by-step reasoning, correctness, and clarity.
    """
    
    # Step 1: PLAN
    plan_prompt = f"""
    You are Study Buddy, a helpful and precise math tutor.
    Your task is to break down the following math problem into a clear, step-by-step plan:
    
    Problem: {query}
    
    Return only the plan as a numbered list, with each step describing the action to be taken.
    Do not solve the problem yet.
    """
    plan_messages = [{"role": "system", "content": plan_prompt}, *context]
    plan = call_openai(plan_messages)

    # Step 2: EXECUTE
    execute_prompt = f"""
    You are Study Buddy, a math tutor.
    Use the plan below to solve the math problem step by step, showing all intermediate steps, formulas, and simplifications.
    Use proper math notation when needed.
    
    Problem: {query}
    Plan:
    {plan}
    
    Now solve the problem in detail.
    """
    execute_messages = [{"role": "system", "content": execute_prompt}, *context]
    execution = call_openai(execute_messages)

    # Step 3: REFINE
    refined_prompt = f"""
    You are Study Buddy, reviewing your own math solution like a peer tutor.
    
    Please carefully review the solution below for:
    - Mathematical correctness
    - Logical flow
    - Clarity and formatting

    If there are any errors or unclear parts, correct and improve them.
    
    Solution to review:
    {execution}
    
    Provide the final, corrected, and clearly formatted solution.
    """
    refined_messages = [{"role": "system", "content": refined_prompt}, *context]
    refined_solution = call_openai(refined_messages)

    return f" Plan:\n{plan.strip()}\n\n Solution:\n{refined_solution.strip()}"

def self_correct_response(query, initial_response, context):
    """Detect and correct errors in a response."""
     
    # Step 1: Detect errors
    detect_prompt = f"""
    You are Study Buddy, a peer tutor. Review this response for errors (factual, logical, or computational):
    Query: {query}
    Response: {initial_response}
    If errors are found, explain them. If none, say "No errors detected."
    """
    detect_messages = [{"role": "system", "content": detect_prompt}, *context]
    error_report = call_openai(detect_messages)
    
    if "no errors detected" in error_report.lower():
        return initial_response
    
    # Step 2: Correct errors
    correct_prompt = f"""
    Based on this error report, provide a corrected response to the query:
    Query: {query}
    Original Response: {initial_response}
    Error Report: {error_report}
    """
    correct_messages = [{"role": "system", "content": correct_prompt}, *context]
    corrected_response = call_openai(correct_messages)
    
    return corrected_response

def fallback_strategy(query, context):
    # Fallback: General LLM knowledge
    fallback_prompt = f"""
    You are Study Buddy, a peer tutor. The primary method failed. Answer {query} using general knowledge or suggest an alternative approach.
    """
    fallback_messages = [{"role": "system", "content": fallback_prompt}, *context]
    return call_openai(fallback_messages)

def llm_confidence_score(query: str, response: str, context) -> float:
    prompt = f"""Evaluate the response to the following question.

        Question: "{query}"
        Response: "{response}"

        Score it from 0 (poor) to 1 (excellent) based on relevance, clarity, completeness, and factual accuracy.
        Respond with just the number.
        """
    messages = [{"role": "system", "content": prompt}, *context]
    score_str = call_openai(messages)
    try:
        score = min(max(float(score_str), 0.0), 1.0)
        return score
    except:
        return None
    
def heuristic_confidence_score(query: str, response: str) -> dict:
    scores = {}

    # Heuristic 1: Length score
    length_score = min(len(response.split()) / 50, 0.3)
    scores["length"] = round(length_score, 3)

    # Heuristic 2: Keyword overlap
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    overlap = len(query_words & response_words) / max(len(query_words), 1)
    scores["overlap"] = round(overlap * 0.2, 3)

    # Heuristic 3: Semantic similarity
    q_embed = embed_model.encode(query, convert_to_tensor=True)
    r_embed = embed_model.encode(response, convert_to_tensor=True)
    sim = util.cos_sim(q_embed, r_embed).item()
    scores["semantic_similarity"] = round(min(sim * 0.3, 0.3), 3)

    # Heuristic 4: Error avoidance
    error_terms = ["error", "not found", "undefined", "can't", "failed"]
    if not any(term in response.lower() for term in error_terms):
        scores["no_error_terms"] = 0.2
    else:
        scores["no_error_terms"] = 0.0

    scores["heuristic_total"] = round(sum(scores.values()), 3)
    return scores

def evaluate_confidence(query: str, response: str, context: list[dict], use_llm: bool = True) -> dict:
    scores = heuristic_confidence_score(query, response)
    
    if use_llm:
        llm_score = llm_confidence_score(query, response, context)
        scores["llm_score"] = round(llm_score, 3) if llm_score is not None else "LLM error"
        
        if isinstance(llm_score, float):
            final = 0.5 * scores["heuristic_total"] + 0.5 * llm_score
        else:
            final = scores["heuristic_total"]
    else:
        scores["llm_score"] = "Skipped"
        final = scores["heuristic_total"]

    scores["final_confidence"] = round(min(final, 1.0), 3)
    return scores
