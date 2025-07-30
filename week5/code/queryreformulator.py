from typing import List, Dict
from call_llm import call_openai
from memorymanager import MemoryManager

class QueryReformulator:
    """Handles query analysis and reformulation when needed"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    async def should_reformulate(self, query: str) -> bool:
        """Determine if query needs reformulation"""

        prompt = f"""
        Analyze if the following query is clear and self-explanatory or if it's vague and needs chat history based reformulation:
        
        Query: {query}
        
        Respond with:
        - "CLEAR" if the query is self-explanatory
        - "VAGUE" if it needs reformulation with context
        """
        messages = [{"role" : "system", "content" : prompt}]
        response = await call_openai(messages)
        return response == "VAGUE"
    
    async def reformulate_query(self, query: str) -> str:
        """Reformulate vague query with relevant chat history"""

        chat_history = self.memory_manager.get_optimized_history()

        prompt = f"""
        Reformulate the vague query using available context and chat history:
        
        Original Query: {query}
        Recent Chat History: {chat_history}
        
        Provide a clear, specific reformulated query that incorporates relevant context.
        """
        messages = [{"role" : "system", "content" : prompt}]
        return await call_openai(messages)
