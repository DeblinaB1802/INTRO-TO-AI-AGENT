from call_llm import call_openai
from sentence_transformers import SentenceTransformer, util

class ConfidenceEvaluator:
    """Evaluates confidence in tool responses"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def llm_confidence_score(self, query: str, response: str) -> float:
        """Use an LLM to rate the response's quality from 0 to 1 based on relevance and accuracy."""

        prompt = f"""Evaluate the response to the following question.

            Question: "{query}"
            Response: "{response}"

            Score it from 0 (poor) to 1 (excellent) based on relevance, clarity, completeness, and factual accuracy.
            Respond with just the number.
            """
        messages = [{"role": "system", "content": prompt}]
        score_str = call_openai(messages)
        try:
            score = min(max(float(score_str), 0.0), 1.0)
            return score
        except:
            return 0.0

    def heuristic_confidence_score(self, query: str, response: str) -> dict:
        """Compute heuristic confidence score using length, keyword overlap, semantic similarity, and error terms."""

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
        q_embed = self.embed_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        r_embed = self.embed_model.encode(response, convert_to_tensor=True, show_progress_bar=False)
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

    def evaluate_confidence(self, query: str, response: str, use_llm: bool = True) -> dict:
        """Combine heuristic and LLM scores to compute a final confidence score for the response."""
        scores = self.heuristic_confidence_score(query, response)
        
        if use_llm:
            llm_score = self.llm_confidence_score(query, response)
            scores["llm_score"] = round(llm_score, 3) if llm_score is not None else "LLM error"
            
            if isinstance(llm_score, float):
                final = 0.5 * scores["heuristic_total"] + 0.5 * llm_score
            else:
                final = scores["heuristic_total"]
        else:
            scores["llm_score"] = "Skipped"
            final = scores["heuristic_total"]

        scores["final_confidence"] = round(min(final, 1.0), 3)
        return scores["final_confidence"]
    

class SelfCorrector:
    """Use a LLM to identify error in its own response and self-correct it if found an error"""

    def self_correct_response(self, query: str, initial_response: str, style: str) -> str:
        """Detect and correct errors in a response."""
        
        # Step 1: Detect errors
        detect_prompt = f"""
        You are Study Buddy, a peer tutor. Review this response for errors (factual, logical, or computational):
        Query: {query}
        Response: {initial_response}
        If errors are found, explain them. If none, say "No errors detected."
        """
        detect_messages = [{"role": "system", "content": detect_prompt}]
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
        correct_messages = [{"role": "system", "content": correct_prompt}]
        corrected_response = call_openai(correct_messages, style)
        
        return corrected_response
