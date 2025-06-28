# 🤖 Study Buddy — Week 3 Prototype

This repository contains the **Week 3 version** of the Study Buddy AI Agent from the "Intro to AI Agents" project. This version focuses on a simplified question-answering pipeline where user queries are classified and routed through distinct logic paths such as retrieval-based search, math reasoning, fallback responses, or LLM-based general answering.

---

## 🎯 Objective

To build an interactive CLI-based agent that:
- Accepts user questions
- Classifies the type of question (retrieval-based, math, general, etc.)
- Uses either internal notes, external search, or LLMs to generate relevant answers
- Stores interaction history for context
- Applies basic error handling and fallback logic

---

## 🧠 Supported Workflows

The system supports multiple paths depending on the query type:

### 1. Retrieval-Based QA
- Fetches relevant chunks from preloaded notes
- Constructs prompt with notes + question
- Sends prompt to OpenAI for answer generation

### 2. Math Reasoning
- Directs math-related questions to a dedicated reasoning function
- Handles computation or formulaic queries

### 3. External Search (if implemented)
- Uses tools like Wikipedia or web search (in future extension)

### 4. Fallback
- If the system cannot determine the query type or retrieval fails, a fallback prompt is used with LLM knowledge

---

## 💬 Example Session

```bash
Welcome to Study Buddy!

Enter your question (or type 'exit' to quit): What is Newton's second law?

Classified query type: retrieval

***ANSWER***
Newton’s second law states that the force acting on an object is equal to the mass of the object times its acceleration (F = ma).

---

Enter your question (or type 'exit' to quit): What is 12 * (5 + 3)?

Classified query type: math

***ANSWER***
The answer is 96.

---

Enter your question (or type 'exit' to quit): Who won the world cup in 2018?

Classified query type: fallback

***ANSWER***
France won the 2018 FIFA World Cup, defeating Croatia in the final.

---

Enter your question (or type 'exit' to quit): exit
Good Bye!
```

---

## 📁 Files

- `study_buddy_week3.py` — Main Python script implementing the CLI-based QA system

---

## 🚀 How to Run

1. Clone this repo:
```bash
git clone https://github.com/DeblinaB1802/INTRO-TO-AI-AGENT.git
cd INTRO-TO-AI-AGENT/week3
```

2. Install requirements (if any):
```bash
pip install openai
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

4. Run the script:
```bash
python study_buddy_week3.py
```

---

*Happy studying! 📚✨*
