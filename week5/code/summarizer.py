from call_llm import call_openai
from memorymanager import MemoryManager

class Summarizer:
    """Handles session and historical topic summarization"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def summarize_recent_session(self) -> str:
        """Summarize current session activities"""

        chat_history = self.memory_manager.chat_history[-10:] if len(self.memory_manager.chat_history) > 10 else self.memory_manager.chat_history
        session_notes = self.memory_manager.session_notes
        
        prompt = f"""
        Summarize the current learning session based on:
        
        Chat History: {chat_history}
        Session Notes: {session_notes}
        
        Provide a concise summary covering:
        1. Main topics discussed
        2. Key questions asked
        3. Learning progress
        4. Areas of focus
        
        Keep summary under 200 words.
        """
        messages = [{"role" : "system", "content" : prompt}]
        return call_openai(messages)
    
    def summarize_past_topics(self) -> str:
        """Summarize historical learning topics"""
        
        if not self.memory_manager.long_term_notes:
            return "No previous learning history available."
        
        notes_history = self.memory_manager.long_term_notes[-10:] if len(self.memory_manager.long_term_notes) > 10 else self.memory_manager.long_term_notes 
        chat_history = self.memory_manager.chat_history[-50:] if len(self.memory_manager.chat_history) > 50 else self.memory_manager.chat_history

        prompt = f"""
        Analyze the long-term learning history and provide a summary of:
        
        Chat History: {chat_history}
        Notes History: {notes_history}
        
        Include:
        1. Major topics studied over time
        2. Learning patterns and preferences
        3. Knowledge areas covered
        4. Progression in understanding
        
        Keep summary under 500 words.
        """
        messages = [{"role" : "system", "content" : prompt}]
        return call_openai(messages)
