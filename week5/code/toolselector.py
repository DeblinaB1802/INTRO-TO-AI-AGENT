from typing import List
from enum import Enum
import json 
from call_llm import call_openai

class ToolType(Enum):
    RAG = "rag"
    MATH_SOLVER = "math_solver"
    TAVILY_SEARCH = "tavily_search"
    WIKIPEDIA_SEARCH = "wikipedia_search"
    SUMMARIZER = "summarizer"
    QUIZ_GENERATOR = "quiz_generator"
    NOTES_EVALUATOR = "notes_evaluator"
    PLANNER = "planner"
    FOLLOW_UP = "follow_up"
    FALLBACK = "fallback"

class ToolSelector:
    """Selects appropriate tools based on query analysis"""
    
    def __init__(self):
        pass
    
    async def select_tools(self, query: str) -> List[ToolType]:
        """Select appropriate tools for the given query"""

        prompt = f"""
        Analyze the user query and select the most appropriate tool(s) from the list below. Multiple tools may be selected if needed. If the query is unclear or 
        lacks required info (e.g. topic, goal, timeframe), include FOLLOW_UP to ask for clarification.
        Most tool responses should be followed by FOLLOW_UP to guide the conversation naturally. Use FALLBACK only if no tool applies, even after clarification.

        Tools & When to Use
        "RAG" : For questions about content in user-provided notes or session data.
        "MATH_SOLVER" : For math problems, formulas, or equations.
        "TAVILY_SEARCH" : For current events, real-time, or web-based info.
        "WIKIPEDIA_SEARCH" : For general factual or encyclopedic queries.
        "SUMMARIZER" : When the user asks to summarize past sessions or notes.
        "QUIZ_GENERATOR" : For generating quizzes. Use FOLLOW_UP if topic/session is not specified.
        "NOTES_EVALUATOR" : Only if user explicitly asks to review/improve notes.
        "PLANNER" : For study plans. If vague (missing goal, subject, time), use with FOLLOW_UP.
        "FOLLOW_UP" : When query is unclear, missing context, or needs disambiguation. Use with tools when appropriate.
        "FALLBACK" : Only when no tool applies, even after FOLLOW_UP.

        Rules
        Select multiple tools if needed.
        If intent is somewhat clear but missing details, pair tool(s) with FOLLOW_UP.
        Prefer FOLLOW_UP over assuming.
        Do not guess — ask.
        
        Query: {query}
        
        Respond with a JSON list of tool names, e.g., ["rag", "math_solver"]
        """
        messages = [{"role" : "user", "content" : prompt}]
        response = await call_openai(messages)
        try:
            tool_names = json.loads(response.strip())
            return [ToolType(name.lower()) for name in tool_names if name.lower() in [t.value for t in ToolType]]
        except:
            return [ToolType.FALLBACK]
