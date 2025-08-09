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
    
    def select_tools(self, query: str) -> List[ToolType]:
        """Select appropriate tools for the given query"""

        prompt = f"""
        You are a tool selector that analyzes the user's query and determines the most appropriate tool(s) from the list below. 

        Key Guidelines:
        - include "FOLLOW_UP" along with other tools unless the user interaction experience might enhance with follow up.
        - If the query involves summarizing, includes terms like "summary", "summarize", "brief", "overview" on a mentioned topic or past topics or sessions, include the "SUMMARIZER" tool.
        - Use "FALLBACK" only when no tool applies, even after clarification.

        Tool Descriptions:
        - "RAG": Use when the query is about content in user-provided notes or session data.
        - "MATH_SOLVER": For math-related questions, equations, or formulas.
        - "TAVILY_SEARCH": For real-time, current events, or web information.
        - "WIKIPEDIA_SEARCH": For general factual or encyclopedic queries.
        - "SUMMARIZER": When the query requests summaries, overviews, or briefing about sessions, topics, or notes.
        - "QUIZ_GENERATOR": When asked to generate quizzes. Include FOLLOW_UP if necessary
        - "NOTES_EVALUATOR": Only if the user explicitly asks to review or improve notes.
        - "PLANNER": For study or learning plans. If the query lacks subject, goal, or timeframe, use FOLLOW_UP with it.
        - "FOLLOW_UP": Include this whenever the query lacks clarity or full context. Use with tools as needed.
        - "FALLBACK": Use only when no tool applies, even after asking clarifying questions.

        Rules:
        - Prefer FOLLOW_UP over making assumptions.
        - Select multiple tools if more than one is applicable.
        - DO NOT guess the user's intent — request clarification using follow up when in doubt.

        Query: {query}

        Respond with a JSON list of lowercase tool names, e.g., ["summarizer", "follow_up"]
        """

        messages = [{"role" : "user", "content" : prompt}]
        response = call_openai(messages)
        try:
            tool_names = json.loads(response.strip())
            return [ToolType(name.lower()) for name in tool_names if name.lower() in [t.value for t in ToolType]]
        except:
            return [ToolType.FALLBACK]
