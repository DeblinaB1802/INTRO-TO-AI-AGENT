
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




