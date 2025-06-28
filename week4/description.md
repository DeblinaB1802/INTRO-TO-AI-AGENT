
# 🧠 Study Buddy — An Interactive Retrieval-Augmented AI Agent

**Study Buddy** is a Retrieval-Augmented Generation (RAG)-based AI assistant designed to help students engage with their study material intelligently. Instead of generating answers from scratch, it smartly searches your notes and delivers relevant, context-rich answers based on what *you’ve already studied*.

This repository is part of a broader AI Agents initiative, focusing on **RAG pipelines, prompt engineering, fallback mechanisms, and confidence-aware LLM responses**.

---

## 🎯 Project Objective

The primary goal is to build an interactive question-answering agent that:
- Retrieves relevant information from your **personal notes**
- Uses LLMs like OpenAI's GPT to construct high-quality, context-aware answers
- Handles a wide variety of questions: direct fact-based, reasoning-based, or general knowledge
- Incorporates ethical and fallback mechanisms to ensure safe and accurate responses
- Scores its own responses using both **heuristic** and **LLM-based** confidence estimation

---

## 💡 Core Features

### 📄 PDF-to-Vector Indexing
- Notes (PDFs) placed in a directory are automatically:
  - Loaded using `PDFPlumber`
  - Split into chunks using `RecursiveCharacterTextSplitter`
  - Embedded via OpenAI Embeddings API
  - Stored in a vector store using **ChromaDB**

### 🧠 Multi-Strategy Query Routing
- Questions are classified into types such as:
  - `"rag"` → Uses your notes
  - `"math"` → Uses structured reasoning
  - `"wikipedia"` / `"tavily"` → Uses external search
  - `"fallback"` → Uses LLM’s general knowledge

### 🔍 Retrieval-Augmented Generation (RAG)
- When a user asks a question, the system:
  - Embeds the query
  - Searches the vector store for top-k similar chunks
  - Constructs a prompt that includes both question + retrieved notes
  - Sends this to OpenAI's Chat API for answer generation
    
### ✅ Confidence Evaluation
- After a response is generated, the system:
  - Computes a **heuristic confidence score** using:
    - Length, keyword overlap, semantic similarity, and presence of errors
  - Optionally asks the **LLM to rate itself** on accuracy and clarity
  - Combines both into a final confidence score

### 🛡️ Ethical Guardrails
- Every user query and LLM response is checked for:
  - Harmful content
  - Ethically unsafe questions
- Users are warned or blocked when necessary

### 🔁 Self-Correction
- If the confidence score is low:
  - The system attempts to **self-correct** the response using a refined prompt
  - If that fails, it falls back to general LLM knowledge

---

## 📁 Repository Structure

```
code/
├── main.py # Main CLI app for interactive Q&A
├── rag.py # Document chunking, embedding, and retrieval
├── call_llm.py # Unified interface to call OpenAI Chat API
├── classify_query.py # Classifies question type (RAG, math, etc.) and re-structure query
├── chat_history.py # Manages in-memory chat history and summarization
├── prompt_design.py # Builds domain-aware prompts and instructions
├── ethical_guardrail.py # Performs query and response compliance checks
├── reasoning.py # Math reasoning, fallback strategy, self-correction
├── tools.py # Interfaces to Tavily and Wikipedia search
```
---

## 🛠️ How to Set Up

### 1. Clone the Repository
```
git clone https://github.com/DeblinaB1802/INTRO-TO-AI-AGENT.git
cd INTRO-TO-AI-AGENT/week4/code
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```
If requirements.txt is not available, make sure to install:

langchain, chromadb, openai, pdfplumber, sentence-transformers, etc.

### 3. Prepare Your Study Notes

Place all your PDF files in the following folder:
```
notes/
```
The system will automatically parse, chunk, embed, and store them.

### 4. Set Your API Key
You can set your OpenAI and Tavily API key and url using an environment variable or hardcode it in the script (already present for testing):
```
export OPENAI_API_KEY="sk-..."
```
### ▶️ How to Run
Launch the Study Buddy CLI interface:
```
python main.py
```
You'll be prompted to enter your:
Educational level (used to adapt answer style)
Question

Then you'll receive a curated, AI-generated response based on:
Your personal notes
External knowledge (if needed)

---

## 🧭 Study Buddy — Full Execution Flow
### 🔁 Step-by-step Workflow
#### 1. Startup & Initialization
 - `prep_ragdb()` → Loads PDF notes, chunks them, creates embeddings, stores them in Chroma.
 - `embed_domains()` → Embeds domain-specific labels (used if RAG fallback is needed).
 - `load_chat_history()` → Loads prior interactions for context continuity.
 - `get_style_conditioned_prompt(user_level)` → Adapts tone/complexity based on user level (e.g.,      high_school, postgraduate).

#### 2. User Input Loop
Prompt:
```
Enter your question (or type 'exit' to quit):
```
#### 3. Ethical Check (Guardrail)
`check_ethical_compliance(question, query=True)`
→ If unethical, display a warning and skip processing.

#### 4. Context Construction
 - `add_to_history(role="user", content=question)`
 - `extract_entities(question)`

 - `get_optimized_context(question, max_tokens=2000)`
 - → Uses past chat to build context.

 - If empty, uses `summarize_history()` as a fallback.

#### 5. Query Structuring & Classification
 - `structure_query()` → Adds context to rephrase/refine user query.
 - `classify_query()` → Categorizes query into one of:
      - `"rag"`
      - `"math"`
      - `"wikipedia"`
      - `"tavily"`
      - (anything else → fallback)

### 🔄 Branching Logic Based on Query Type
#### - Query Type: `rag`
 - `retrieve_chunks()` → Fetch top-k chunks from vector DB
 - Construct prompt with retrieved notes
 - `call_openai()` → LLM responds based only on notes
 - If keywords like `error`, `not found`, etc. are detected:
 - Build domain-specific prompt using `build_domain_specific_prompt()`
 - Retry `call_openai()` with new prompt

#### - Query Type: `math`
 - `plan_execute_refine_math()` → Handles multi-step math reasoning
 - No RAG or external search involved

#### - Query Type: `wikipedia`
 - `search_wikipedia(structured_query)` → Retrieves Wikipedia summary
 - Prompt LLM with the wiki content
 - call_openai() → Generates response

#### - Query Type: `tavily`
 - `search_tavily(structured_query)` → Uses Tavily API or similar external tool
 - Prompt LLM with the Tavily response
 - `call_openai()` → Generates response

#### - Query Type: Other (Fallback)
 - `fallback_strategy()` → Uses LLM with no external knowledge
 - Prompt is: "Answer using general knowledge..."

### ✅ Post-Answer Evaluation
#### 6. Confidence Evaluation
 - `evaluate_confidence(query, response, context, use_llm=True)`
   → Combines:
 - Heuristic score (based on length, overlap, semantic sim, error terms)
 - LLM self-evaluation score

#### 7. Self-Correction Logic
 - If `final_confidence < 0.7`:
 - Try `self_correct_response()` with rephrased prompt
 - Re-evaluate
 - If still low confidence, use `fallback_strategy()`

#### 8. Logging & History
 - `add_to_history(role="assistant", content=corrected_answer)`
 - Re-check ethical safety of LLM response
 - Display final answer with:
```
***ANSWER***
<final_answer>
```

---

## 🌐 High-Level Execution Paths
Here’s a simplified tree-style flow chart for clarity:

```
Start
 └── load notes, vectors, chat history
      └── user input → ethical check
           └── structure & classify query
                ├── rag → retrieve_chunks → call LLM
                │        └── if failed → domain_prompt → call LLM
                ├── math → plan_execute_refine_math
                ├── wikipedia → search_wikipedia → call LLM
                ├── tavily → search_tavily → call LLM
                └── fallback → call LLM directly
                      ↓
           evaluate_confidence
                └── low? → self-correct → failed → fallback → call LLM directly → save to history + print final answer
                                   └── succeeded → save to history + print final answer
```




