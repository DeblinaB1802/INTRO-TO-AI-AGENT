# 🧠 Study Buddy — Week 5 Implementation (RAG + LangGraph + Custom Agentic Tool Orchestration)

This `week5` implementation builds on Week 4’s RAG-based StudyBuddy and adds a production-oriented orchestration layer using **LangGraph** plus a **custom agentic tool orchestration module** for one-tool-at-a-time execution, self-correction, and confidence-aware retries.  
It’s designed to be modular, testable, and extendable so you can iterate quickly on new tools and guardrails.

---

## 🎯 What’s new in Week 5 (high level)

- **LangGraph orchestration** — main workflow graph for query preprocessing (ethical checks, PDF ingestion, summarization, query reformulation, tool selection) and a **subgraph** that executes each selected tool independently with retry/self-correct loops.
- **Custom Agentic Tool Orchestration Module** — single place where tool implementations live (`mcptools.py`) and a graph-based executor that:
  - Runs tools sequentially, each in its own state cycle.
  - Evaluates confidence, self-corrects if needed, retries if confidence is low.
  - Adds post-response ethical checks for certain tools.
- **Interactive Quiz Engine** — robust quiz generation and evaluation with JSON parsing, fallback questions, evaluation & final results (`quizmaster.py`).
- **Self-correction & Confidence Evaluator** — tools get a confidence score; low-confidence responses are refined and retried.
- **Memory-first design** — `MemoryManager` is used for session notes, chat history, and RAG retrieval.
- **PDF support** — PDF extraction + adding to vector memory (same RAG pipeline as Week 4, but integrated into LangGraph workflow).
- **CLI friendly** — `studybuddy.py` exposes an interactive session loop (educational level → question → optional PDF).

---

## 🛠️ Tech stack & Dependencies

- Python 3.9+
- LangGraph — workflow orchestration
- LangChain (small parts, prompts) — prompt templating
- OpenAI (Chat / Embeddings) — LLM and embeddings (via `call_llm.py`)
- Chroma / any vector DB — for RAG (via `MemoryManager`)
- pdfplumber — PDF text extraction
- scikit-learn — similarity utilities
- dotenv — for environment variables (API keys)

---

## 📁 Repo layout (week5 focus)

```
week5/
├── main.py                  # Main agent (LangGraph graphs + CLI entry)
├── ethicalguardrail.py      # Query/response safety checks
├── memorymanager.py         # Chat history, PDF session notes, vector DB wrapper
├── call_llm.py              # Thin wrapper around OpenAI / LLM calls
├── pdfprocessor.py          # PDF extraction utilities
├── queryreformulator.py     # Query reformulation heuristics
├── toolselector.py          # Maps queries -> ToolType(s)
├── mcptools.py              # Tool implementations (RAG, math, tavily, wiki, planner, etc.)
├── quizmaster.py            # Quiz generation + interactive quiz runner
├── summarizer.py            # session/past summarizer used in graph
├── self_evaluator.py        # Confidence scoring and self-correction routines
├── searchtools.py           # Searching tool like wikipedia or Tavilly for searching from external knowledge base
├── utils.py                 # prompt styles, helper functions
├── README.md                # This file
├── requirements.txt
├── chat_history.py          # Manages and stores past conversation history
├── long_term_notes.py       # Stores long-term notes
├── chroma_db                # Local vector database for storing and retrieving embeddings
├── long_term_notes.py       # Stores and retrieves long-term notes for RAG
└── examples/                # example PDFs, sample sessions, demo scripts
```

---

## ▶️ How to set up (quick)

1. Clone:
```bash
git clone https://github.com/DeblinaB1802/INTRO-TO-AI-AGENT.git
cd INTRO-TO-AI-AGENT/week5
```

2. Create venv & install:
```bash
python -m venv venv
source venv/bin/activate    # mac/linux
pip install -r requirements.txt
```

3. Add API keys to `.env`:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=...
```

4. Run the CLI:
```bash
python main.py
```

---

## 🔍 Week 5 — Execution flow (detailed)

### Main LangGraph (high-level)
1. **Ethical Check** — Validates query against `EthicalGuardrail`; unsafe queries get a reframe prompt and processing stops.
2. **PDF Processing** — If PDF provided, extract text and store in memory.
3. **Summarization** — Create `session_summary` and `past_summary`.
4. **Query Reformulation** — Clarify or rephrase if needed.
5. **Tool Selection** — Map reformulated query to a list of tools (`ToolSelector`).

### Tool-execution Subgraph (per tool)
- **execute_tool** — Call the corresponding function in `mcptools.py`.
- **evaluate_confidence** — Assigns a score via `ConfidenceEvaluator`.
- If low confidence:
  - **self_correct** — Retry with improved prompt.
  - Return to `evaluate_confidence` until accepted or retries exhausted.
- **ethical_guardrail** — For certain tools, check post-response ethics.

---

## 🧩 Custom Agentic Tool Orchestration Module

This module is a **purpose-built execution layer** for StudyBuddy’s educational tools — not a traditional MCP server.  
It’s designed for:
- Sequential execution of multiple tools selected by the main graph.
- Confidence-aware retries and disclaimers.
- Easy extensibility — new tools only need:
  1. Enum entry in `ToolType`
  2. Implementation in `mcptools.py`
  3. Mapping in `studybuddy.py`

This gives you MCP-like flexibility without needing to adhere to an external spec.

---

## 🧩 Tools implemented in `mcptools.py`

| Tool | Description |
|------|-------------|
| **rag_tool** | Retrieves answer from uploaded notes via vector search. |
| **tavily** | Queries Tavily API for fresh info and constrains answer to provided content. |
| **wiki** | Wikipedia search + answer synthesis. |
| **math_solver** | Plan → Execute → Refine math reasoning pipeline. |
| **quiz_generator** | Interactive 10-question descriptive quiz with evaluation. |
| **notes_evaluator** | Reviews session notes for completeness, clarity, and coverage. |
| **planner** | Generates day-by-day study plan from goals and past summaries. |
| **fallback_strategy** | General knowledge answer when other tools fail. |
| **follow_up** | Continues conversation naturally with context awareness. |
| **summarizer** | Returns session or past summaries, or generates new summaries. |

---

## 🧪 Quiz Engine (`quizmaster.py`) features

- Quiz types: `topic`, `session`, `past_topics`
- 10 descriptive questions (4 easy, 4 medium, 2 hard)
- LLM-based evaluation per question (score + feedback)
- Final results + difficulty breakdown + overall feedback

---

## ✅ Confidence evaluation & self-correction

- Combines heuristic + optional LLM self-rating
- Retries low-confidence outputs via `SelfCorrector`
- Adds disclaimer if still low after retries

---

## 🧭 Example CLI session

```
$ python studybuddy.py
Welcome to Study Buddy — your partner in learning.
Enter your educational details: college
Enter your question (or 'exit'): Explain transformer architecture.
Upload your notes pdf: notes/transformers.pdf
...
```

---

## 🧩 Development notes

- **Add new tools**: implement in `mcptools.py`, update `ToolType`, map in `_execute_single_tool_node()`
- **Swap vector DB**: Change in `MemoryManager`
- **Test individual tools**: mock `call_llm` for deterministic tests

---

