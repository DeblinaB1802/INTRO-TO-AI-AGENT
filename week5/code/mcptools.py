from memorymanager import MemoryManager
from typing import Tuple
import json
from langchain.prompts import ChatPromptTemplate
from call_llm import call_openai
from quizmaster import QuizType, QuizGenerator
from searchtools import search_tavily, search_wikipedia
from summarizer import Summarizer

class MCPTools:
    """Implementation of MCP server tools"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    
    def rag_tool(self, query: str, style: str) -> str:
        """RAG tool for information retrieval from provided notes"""

        relevant_docs = self.memory_manager.retrieve_relevant_context(query)
        
        if not relevant_docs:
            return "No relevant information found in notes."
        
        prompt = f"""
        Answer the question based on the provided context from notes:
        
        Question: {query}
        Relevant Documents: {relevant_docs}
        
        Provide a comprehensive answer based on the available information.
        If information is insufficient, clearly state what's missing.
        """
        messages = [{"role" : "system", "content" : prompt}]
        return call_openai(messages, style)
    
    def tavily(self, query: str, style: str) -> str:
        """Tavily tool for recent information retrieval from external source"""

        response = search_tavily(query)
        messages = [
            {"role": "system", "content": f"""You are an AI Study Assistant. Answer the following question ONLY based on the content provided to you. 
            If you can't find the answer in the provided content, state that "I can't find the answer in the provided content." """},
            {"role": "user", "content": f"Notes: {response}\nQuestion: {query}"}
        ]
        return call_openai(messages, style)

    def wiki(self, query: str, style: str) -> str:
        """Wikipedia tool for factual information retrieval from external source"""

        response = search_wikipedia(query)
        messages = [
            {"role": "system", "content": f"""You are an AI Study Assistant. Answer the following question ONLY based on the content provided to you. 
            If you can't find the answer in the provided content, state that "I can't find the answer in the provided content." """},
            {"role": "user", "content": f"Notes: {response}\nQuestion: {query}"}
            ]
        return call_openai(messages, style)
    
    def math_solver(self, query: str, style: str) -> str:
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
        plan_messages = [{"role": "system", "content": plan_prompt}]
        plan = call_openai(plan_messages, style)

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
        execute_messages = [{"role": "system", "content": execute_prompt}]
        execution = call_openai(execute_messages, style)

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
        refined_messages = [{"role": "system", "content": refined_prompt}]
        refined_solution = call_openai(refined_messages, style)

        return f" Plan:\n{plan.strip()}\n\n Solution:\n{refined_solution.strip()}"

    def quiz_generator(self, query: str, session_summary: str, past_summary: str) -> str:
        """Generate quiz based on type and content"""

        prompt = f"""
        You are a study assistant that helps students decide what type of quiz they are asking for.

        The quiz can be about:
        1. A specific topic (label as "topic")
        2. The current session's content (label as "session")
        3. Past topics they’ve studied (label as "past_topics")

        Given the user's query below, determine which type of quiz they are requesting.

        Respond with only one of the following exact labels: topic, session, or past_topics.

        Query: "{query}"
        Answer:
        """
        messages = [{"role" : "system", "content" : prompt}]
        quiz_type = call_openai(messages)

        quizmaster = QuizGenerator()
        try:
            quizmaster.run_interactive_quiz(query, quiz_type, session_summary, past_summary)
            return "Quiz is Over."
        except:
            return "Failed to generate your quiz. Please retry."
        
    def notes_evaluator(self, style: str) -> str:
        """Evaluate notes and suggest improvements"""

        prompt = f"""
        Evaluate the following notes and provide feedback:
        
        Notes Content: {self.memory_manager.session_notes}
        
        Analyze:
        1. Completeness and coverage
        2. Organization and structure
        3. Clarity and readability
        4. Missing important concepts
        5. Suggestions for improvement
        6. Additional relevant topics to explore
        
        Provide constructive feedback and actionable recommendations.
        """
        messages = [{"user" : "system", "content" : prompt}]
        return call_openai(messages, style)
    
    def planner(self, query: str, past_summary: str, style: str) -> str:
        """Create detailed study plan"""
        
        prompt = f"""
        You are an expert learning strategist.

        You will receive:
        1. A user **query** – describing the user's goal and constraints (e.g., target date, study capacity, deadlines).
        2. A **long-term note** – containing the user's knowledge base, learning history, strengths, weaknesses, and preferences.

        ### Your Task:

        1. Extract and clearly list the following:
        - **Goal(s)** – derived from the query
        - **Constraints** – derived from the query (e.g., time span, study days per week)
        - **User Knowledge Base** – derived from the long-term note

        2. Based on these, create a **comprehensive day-by-day study plan**:
        - Assign tasks to each day (e.g., "Read Chapter 1", "Watch Intro Video", "Take Practice Quiz")
        - Do **not assign specific time slots**
        - Each task should be tagged with one of: [Study], [Practice], [Review], [Assessment], or [Rest]
        - Recommend specific resources or tools when appropriate

        3. Format the plan by **calendar date** or **Day X format** based on available constraints.

        4. Ensure the plan is:
        - Achievable within the user's constraints
        - Personalized using the user’s existing knowledge
        - Flexible enough to adapt weekly
        - Easy to copy into Google Calendar (one event per day)

        ### Output Format:

        **Goal(s):**
        - ...

        **Constraints:**
        - ...

        **User Knowledge Base:**
        - ...

        **Study Plan (Daily Schedule):**

        - 📅 Day 1 – July 27, 2025
        - [Study] Read Chapter 1: "Fundamentals of Machine Learning"
        - [Practice] Complete 3 basic ML exercises on Kaggle

        - 📅 Day 2 – July 28, 2025
        - [Review] Summarize notes from Day 1 in a notebook
        - [Study] Watch "Supervised vs Unsupervised Learning" on YouTube (3–5 min)

        - 📅 Day 3 – July 29, 2025
        - [Assessment] Take mini-quiz on foundational ML concepts (via Quizlet)

        ...

        Continue the schedule based on the goal timeline.

        ### Notes:
        - Do not include specific times.
        - Focus on clarity, structure, and realistic pacing.
        - Keep each entry actionable and aligned to the user’s capacity.

        Query: {query}
        Past notes summary: {past_summary}
        """

        messages = [{"role" : "user", "content" : prompt}]
        return call_openai(messages, style)
    
    def fallback_strategy(self, query: str, style: str) -> str:
        """Generate a fallback response using general LLM knowledge when primary methods fail."""

        fallback_prompt = f"""
        You are Study Buddy, a peer tutor. The primary method failed. Answer {query} using general knowledge or suggest an alternative approach.
        """
        fallback_messages = [{"role": "system", "content": fallback_prompt}]
        return call_openai(fallback_messages, style)
    
    def follow_up(self, query: str, style: str) -> str:
        """
        Generate a follow-up response in an ongoing tutoring conversation using general LLM capabilities.
        """
        chat_history = self.memory_manager.chat_history[-10] if len(self.memory_manager.chat_history) > 10 else self.memory_manager.chat_history

        followup_prompt = f"""
        You are Study Buddy, a friendly and knowledgeable peer tutor helping a student through an ongoing conversation.
        
        Conversation so far:
        {chat_history}

        Your role:
        - Reply naturally and informatively to the student's latest message.
        - Keep a warm, peer-like tone.
        - Explain concepts clearly with examples or analogies if helpful.
        - Clarify confusion or rephrase when needed.
        - Encourage learning with follow-up questions or suggestions.
        - Be concise but thorough.

        Now, respond to the student’s latest message: {query}
        """
        
        followup_messages = [{"role": "system", "content": followup_prompt}]
        
        return call_openai(followup_messages, style)

    def summarizer(self, query: str, session_summary: str, past_summary: str, style: str) -> str:
        """
        Returns a session summary, past summary, or generates a new one based on the query.
        """

        prompt = f"""
        A user asked: "{query}"

        Decide what kind of summary the user is asking for:
        - If it's about the current session, respond with "session".
        - If it's about past sessions or previous topics, respond with "past".
        - If it's about a completely new topic not covered in either, respond with "other".

        Respond with only one word: "session", "past", or "other".
        """
        messages = [{"role": "user", "content": prompt}]
        response = (call_openai(messages)).strip().lower()

        if response == "session":
            return session_summary
        elif response == "past":
            return past_summary
        elif response == "other":
            summary_prompt = f"""
            You are a summarizer.

            The user asked: "{query}"
            
            Generate a clear and brief summary based on their request.
            Return only the summary text.
            """
            messages = [{"role": "user", "content": summary_prompt}]
            return call_openai(messages, style)
        else:
            return "I'm not sure what you're referring to. Could you clarify your request?"
