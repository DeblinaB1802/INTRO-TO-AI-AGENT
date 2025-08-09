from typing import Tuple, Optional
from call_llm import call_openai

class EthicalGuardrail:
    """Handles ethical content filtering and query validation"""
    
    def __init__(self):
        self.sensitive_topics = [
            "violence", "self-harm", "illegal activities", "hate speech",
            "dangerous instructions", "personal information extraction"
        ]
    
    def check_query_ethics(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if query passes ethical guidelines"""

        prompt = f"""
        Analyze the following query for ethical concerns. Check for:
        1. Requests for harmful, illegal, or dangerous content
        2. Attempts to extract personal information
        3. Hate speech or discriminatory content
        4. Self-harm or violence-related content
        
        Sensitive topics: {self.sensitive_topics}
        Query: {query}
        
        Respond with:
        - "PASS" if the query is ethically acceptable
        - "FAIL: [reason]" if it violates guidelines
        - "REFRAME: [suggestion]" if it needs minor adjustments
        """
        messages = [{"role" : "system", "content" : prompt}]
        response = call_openai(messages)
        result = response.strip()
        
        if result.startswith("PASS"):
            return True, None
        elif result.startswith("REFRAME:"):
            return False, result[9:].strip()
        else:
            return False, result[5:].strip() if result.startswith("FAIL:") else result
    
    def check_response_ethics(self, response: str) -> Tuple[bool, str]:
        """Filter potentially sensitive information from responses"""
        
        prompt = f"""
        Review the following response for sensitive information that should be filtered:
        1. Personal identifiable information
        2. Harmful or dangerous content
        3. Inappropriate content for educational context
        
        Sensitive topics: {self.sensitive_topics}
        Response: {response}
        
        If content needs filtering, provide a cleaned version.
        If content is appropriate, respond with: "APPROVED: [original response]"
        If content should be blocked, respond with: "BLOCKED: [reason]"
        """
        
        messages = [{"role" : "system", "content" : prompt}]
        response = call_openai(messages)
        content = response.strip()
        
        if content.startswith("APPROVED:"):
            return True, content[9:].strip()
        elif content.startswith("BLOCKED:"):
            return False, "I cannot provide this information due to content policy restrictions."
        else:
            return True, content  # Filtered version