"""
RAG-based Chatbot Agent with LangGraph and MCP Server
A comprehensive educational assistant with ethical guardrails, memory management, and multiple tools.
"""
import logging
from datetime import datetime, date
from typing import Dict, List, Optional,Any
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END, START
from sklearn.metrics.pairwise import cosine_similarity
from ethicalguardrail import EthicalGuardrail
from toolselector import ToolSelector, ToolType
from summarizer import Summarizer
from queryreformulator import QueryReformulator
from memorymanager import MemoryManager
from mcptools import MCPTools
from call_llm import call_openai
from self_evaluator import ConfidenceEvaluator, SelfCorrector
from utils import PDFProcessor, get_style_conditioned_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatState:
    style: str
    query: str
    pdf_content: Optional[str] = ""
    ethical_check_passed: bool = False
    reformulated_query: Optional[str] = ""
    selected_tools: List[ToolType] = field(default_factory=list)
    retry_tools: List[ToolType] = field(default_factory=list)
    tool_responses: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    final_response: str = ""
    chat_history: List[Dict] = field(default_factory=list)
    session_summary: str = ""
    past_summary: str = ""
    retry_count: Dict[str, int] = field(default_factory=dict)
    retry: bool = False
    current_tool: Optional[ToolType] = None

class StudyBuddy:
    """Main chatbot agent orchestrating all components"""
    
    def __init__(self):

        self.ethical_guardrail = EthicalGuardrail()
        self.memory_manager = MemoryManager()
        self.summarizer = Summarizer(self.memory_manager)
        self.query_reformulator = QueryReformulator(self.memory_manager)
        self.tool_selector = ToolSelector()
        self.mcp_tools = MCPTools(self.memory_manager)
        self.confidence_evaluator = ConfidenceEvaluator()
        self.self_evaluator = SelfCorrector()
        self.pdf_reader = PDFProcessor()
        
        # Initialize LangGraph
        self.graph = self._build_graph()
        self.subgraph = self._build_subgraph()
        #self.checkpointer = SqliteSaver.from_conn_string(":memory:")

    def process_query(self, query: str, pdf_path: Optional[str] = None, style: str = "high_school") -> str:
        """Main entry point for processing queries"""
        if pdf_path:
            pdf_content = self.pdf_reader.extract_text_from_pdf(pdf_path)
        else:
            pdf_content = ""
        self.memory_manager.add_to_chat_history({"role" : "user", "content" : query})
        state = ChatState(
            style=style,
            query=query,
            pdf_content=pdf_content,
            chat_history=self.memory_manager.chat_history.copy()
        )
        
        # Run the graph
        output_dict = self.graph.invoke(state)
        state = ChatState(**output_dict)

        print(f"Tools to be executed: {len(state.selected_tools)}")
        print(state.selected_tools)
        
        for i, tool in enumerate(state.selected_tools):
            state.current_tool = tool

            output_dict = self.subgraph.invoke(state)
            print(output_dict['tool_responses'][tool.value])

            if tool in [ToolType.RAG, ToolType.MATH_SOLVER, ToolType.TAVILY_SEARCH, ToolType.WIKIPEDIA_SEARCH, ToolType.FALLBACK, ToolType.FOLLOW_UP, ToolType.PLANNER, ToolType.NOTES_EVALUATOR]:
                self.memory_manager.add_to_chat_history({"role" : "system", "content" : output_dict['tool_responses'][tool.value]})

            state = ChatState(**output_dict)
    
    def _ethical_check_node(self, state: ChatState) -> ChatState:
        """Check ethical guidelines"""

        passed, suggestion = self.ethical_guardrail.check_query_ethics(state.query)
        state.ethical_check_passed = passed
        if not passed:
            state.final_response = f"Please reframe your query: {suggestion}"

        return state
    
    def _process_pdf_node(self, state: ChatState) -> ChatState:
        """Process PDF if provided"""

        if state.pdf_content:
            session_id = str(datetime.now().isoformat())
            self.memory_manager.add_pdf_to_memory(
            state.pdf_content,
            session_id
        )
        return state
    
    def _summarize_node(self, state: ChatState) -> ChatState:
        """Generate session and past summaries"""

        state.session_summary = self.summarizer.summarize_recent_session(state.style)
        state.past_summary = self.summarizer.summarize_past_topics(state.style)
        return state
    
    def _reformulate_query_node(self, state: ChatState) -> ChatState:
        """Reformulate query if needed"""

        if self.query_reformulator.should_reformulate(state.query):
            state.reformulated_query = self.query_reformulator.reformulate_query(state.query)
        else:
            state.reformulated_query = state.query
        return state

    def _select_tools_node(self, state: ChatState) -> ChatState:
        """Select appropriate tools"""
        state.selected_tools = self.tool_selector.select_tools(state.reformulated_query)
        return state
   
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("ethical_check", self._ethical_check_node)
        workflow.add_node("process_pdf", self._process_pdf_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("reformulate_query", self._reformulate_query_node)
        workflow.add_node("select_tools", self._select_tools_node)
        
        # Add edges
        workflow.add_edge(START, "ethical_check")

        workflow.add_conditional_edges(
            "ethical_check",
            lambda state: "reframe" if not state.ethical_check_passed else "continue",
            {"reframe": END, "continue": "process_pdf"}
        )
        workflow.add_edge("process_pdf", "summarize")
        workflow.add_edge("summarize", "reformulate_query")
        workflow.add_edge("reformulate_query", "select_tools")
        workflow.add_edge("select_tools", END)
        
        return workflow.compile()
    

    def _execute_single_tool_node(self, state: ChatState) -> ChatState:
        """Execute selected tool only one at a time"""
        tool = state.current_tool
        try:
            if tool == ToolType.RAG:
                response = self.mcp_tools.rag_tool(
                    state.reformulated_query, state.style
                )
                state.tool_responses[tool.value] = response
            
            elif tool == ToolType.MATH_SOLVER:
                response = self.mcp_tools.math_solver(
                    state.reformulated_query
                )
                state.tool_responses[tool.value] = response
            
            elif tool == ToolType.QUIZ_GENERATOR:
                response = self.mcp_tools.quiz_generator(
                    state.reformulated_query, state.session_summary, state.past_summary
                )
                state.tool_responses[tool.value] = response

            elif tool == ToolType.SUMMARIZER:
                response = self.mcp_tools.summarizer(
                    state.reformulated_query, state.session_summary, state.past_summary, state.style
                )
                state.tool_responses[tool.value] = response
            
            elif tool == ToolType.NOTES_EVALUATOR:
                if state.pdf_content:
                    response = self.mcp_tools.notes_evaluator(
                        state.style
                    )
                    state.tool_responses[tool.value] = response
            
            elif tool == ToolType.PLANNER:
                response = self.mcp_tools.planner(
                    state.reformulated_query, state.past_summary, state.style
                )
                state.tool_responses[tool.value] = response
            
            elif tool == ToolType.TAVILY_SEARCH:
                response = self.mcp_tools.tavily(
                    state.reformulated_query, state.style
                )
                state.tool_responses[tool.value] = response

            elif tool == ToolType.WIKIPEDIA_SEARCH:
                response = self.mcp_tools.wiki(
                    state.reformulated_query, state.style
                )
                state.tool_responses[tool.value] = response

            elif tool == ToolType.FOLLOW_UP:
                response = self.mcp_tools.follow_up(
                    state.reformulated_query, state.style
                )
                state.tool_responses[tool.value] = response
            
            elif tool == ToolType.FALLBACK:
                response = self.mcp_tools.fallback_strategy(
                    state.reformulated_query, state.style
                )
                state.tool_responses[tool.value] = response
                
        except Exception as e:
            logger.error(f"Error executing tool {tool.value}: {str(e)}")
            state.tool_responses[tool.value] = f"Error executing {tool.value}: {str(e)}"

        return state

    def _self_correct_single_node(self, state: ChatState) -> ChatState:
        """Self-correct generated responses"""

        tool = state.current_tool
        if tool in [ToolType.RAG, ToolType.TAVILY_SEARCH, ToolType.WIKIPEDIA_SEARCH, ToolType.FALLBACK, 
                    ToolType.PLANNER, ToolType.NOTES_EVALUATOR, ToolType.FOLLOW_UP]:
            corrected_response = self.self_evaluator.self_correct_response(state.reformulated_query, state.tool_responses[tool.value], state.style)
            state.tool_responses[tool.value] = corrected_response
        return state
    
    def _evaluate_confidence_single_node(self, state: ChatState) -> ChatState:
        """Evaluate confidence in responses"""

        tool = state.current_tool
        if tool in [ToolType.RAG, ToolType.TAVILY_SEARCH, ToolType.WIKIPEDIA_SEARCH, ToolType.FALLBACK, 
                    ToolType.PLANNER, ToolType.NOTES_EVALUATOR, ToolType.FOLLOW_UP]:
            confidence_score = self.confidence_evaluator.evaluate_confidence(state.reformulated_query, state.tool_responses[tool.value])
            state.confidence_scores[tool.value] = confidence_score

        return state
    
    def _check_if_retry_single_tool_node(self, state: ChatState) -> bool:
        """Determine if tools should be retried based on confidence. Retry tools with low confidence"""

        tool = state.current_tool
        if tool in [ToolType.RAG, ToolType.TAVILY_SEARCH, ToolType.WIKIPEDIA_SEARCH, ToolType.FALLBACK, 
                    ToolType.PLANNER, ToolType.NOTES_EVALUATOR, ToolType.FOLLOW_UP]:
            if tool.value not in state.retry_count:
                state.retry_count[tool.value] = 0

            score = state.confidence_scores.get(tool.value, 1.0)
            if state.retry_count[tool.value] >= 1 and score < self.confidence_evaluator.confidence_threshold:
                state.tool_responses[tool.value] += f"\n\n*Note: I'm not fully confident about the {tool.value} results. Please verify the information.*"
                return state
            
            if score < self.confidence_evaluator.confidence_threshold:
                state.retry_count[tool.value] += 1
                return True

        return False
    
    def _check_response_ethical_guardrail(self, state: ChatState) -> ChatState:
        """Checks Ethical Compliance for tool responses where required"""

        tool = state.current_tool
        if tool in [ToolType.TAVILY_SEARCH, ToolType.FALLBACK]:
            _, response = self.ethical_guardrail.check_response_ethics(state.tool_responses[tool.value])
            state.final_response = response
        return state
            
    def _build_subgraph(self) -> StateGraph:
        """Build the LangGraph subworkflow for execution of each tool separately"""

        subworkflow = StateGraph(ChatState)

        subworkflow.add_node("execute_tool", self._execute_single_tool_node)
        subworkflow.add_node("self_correct", self._self_correct_single_node)
        subworkflow.add_node("evaluate_confidence", self._evaluate_confidence_single_node)
        subworkflow.add_node("ethical_guardrail", self._check_response_ethical_guardrail)
        
        subworkflow.add_edge(START, "execute_tool")
        subworkflow.add_edge("execute_tool", "evaluate_confidence")

        subworkflow.add_conditional_edges(
            "evaluate_confidence",
            self._check_if_retry_single_tool_node,  # Must return a bool
            {
                True: "self_correct",
                False: "ethical_guardrail"
            }
        )

        subworkflow.add_edge("self_correct", "evaluate_confidence")
        subworkflow.add_edge("ethical_guardrail", END)

        return subworkflow.compile()

def main():

    print("Welcome to Study Buddy — your partner in learning. Let’s explore knowledge together!")
    print("\n\nJust a moment! Your Study Buddy is getting everything ready.")

    studybuddy = StudyBuddy()
    #studybuddy.memory_manager.initialize_vector_store()

    print("\n\nAll set! Your Study Buddy is now ready to evasist you.")

    user_level = input("\nEnter your educational details: (choose from ['middle_school', 'high_school', 'college', 'postgraduate', 'expert'] :")
    style = get_style_conditioned_prompt(user_level)


    while True:

        question = input("\nEnter your question (or type 'exit' to quit):")
        if question.lower() == "exit":
            print("Good Bye!!!")
            break
        pdf_path = input("\nUpload your notes pdf (one at a time): ")
        studybuddy.process_query(question, pdf_path, style)


if __name__ == "__main__":
    main()
