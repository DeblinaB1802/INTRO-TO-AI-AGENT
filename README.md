# INTRO-TO-AI-AGENT
SEASONS OF CODE, IIT BOMBAY

# 🤖 AI Study Buddy Project – My Journey in Intro to AI Agents (WnCC SoC25)

Hi! I'm **Deblina Biswas**, and this repository documents my 8-week learning journey through the **AI Study Buddy Project**, a hands-on curriculum organized by [WnCC Summer of Code](https://wncc-soc.tech-iitb.org).

Throughout the project, I explored the world of **Agentic AI Systems** — combining retrieval-augmented generation (RAG), memory, planning, tool use, and personalization to build an AI chatbot that can assist with studies in an intelligent, interactive way.

---

## 🧭 Project Summary

The aim was to build a **personalized AI-powered Study Buddy**, capable of:
- Answering questions from documents using **RAG**
- Maintaining **short-term and long-term memory**
- Using **external tools** like Wikipedia search
- Conducting **multi-turn conversations** with context awareness
- Making plans and decisions through **agentic autonomy**
- Collecting feedback and maintaining safety via **guardrails**
- Providing a basic **frontend UI** for real-world interaction

Each folder in this repository (`week1`, `week2`, ..., `week8`) contains my weekly progress — including assignments, mini-projects, and notes.

---
# Study Buddy Project – Weekly Progress Overview

**Introduction**  
This repository documents my week-by-week progress on the **Study Buddy Project**, part of the **SOC25 initiative** at IIT Bombay. The goal is to develop an AI-powered Study Buddy chatbot capable of retrieving relevant information, integrating tools, maintaining memory, and reasoning over complex queries. The work follows the 8-week curriculum outlined in the original project plan, with my updates below reflecting progress up to **Week 5**.

---

## Weekly Progress

### **Week 1 – Foundation**
- Understood the basics of AI agents and the core architecture of a conversational agent.
- Set up the development environment with Python, LangChain, and necessary libraries.
- Explored the principles of prompt engineering and basic LLM query handling.

---

### **Week 2 – Retrieval-Augmented Generation (RAG) & Vector Databases**
- Implemented a document ingestion pipeline for course-related notes and materials.
- Created embeddings using OpenAI’s embedding models and stored them in a FAISS vector database.
- Integrated the vector database with LangChain to enable semantic retrieval of documents.
- Tested retrieval with sample queries and verified relevance scores.

---

### **Week 3 – Tools & Memory**
- Integrated external tools (search, calculation, and document reading) into the agent workflow.
- Implemented short-term memory for conversational context using LangChain’s `ConversationBufferMemory`.
- Researched and experimented with long-term memory approaches (vector store-based memory).
- Improved the agent’s ability to maintain context across multiple turns.

---

### **Week 4 – Reasoning & Multi-turn Context-Aware Agents**
- Studied methods for maintaining context across multiple turns in conversation, including state management, entity tracking, and summarization to handle long inputs.
- Implemented tool chaining so the agent can execute multiple tools in sequence, handle dependencies, and recover from errors.
- Applied persona engineering to shape the assistant’s tone, style, and domain specificity while embedding ethical safeguards.
- Experimented with reasoning frameworks such as Plan-Execute-Refine and self-correction mechanisms to improve decision-making.

---

### **Week 5 – LangGraph Integration & LangChain Transition**
- Learned the differences between LangChain and LangGraph, noting LangChain’s usefulness for prototyping but limitations in complex, stateful, or multi-agent workflows.
- Adopted LangGraph for its graph-based architecture, enabling nonlinear workflows, better state retention, and more granular control over agent actions.
- Explored LangGraph’s strengths in multi-agent orchestration, visual debugging via LangGraph Studio, and production scalability.
- Began refactoring the Study Buddy’s architecture from linear LangChain pipelines into a modular LangGraph node-based workflow for better maintainability and flexibility.

---

## 📂 Repository Structure

```
├── week1/ # Prompt-based Q&A system
├── week2/ # RAG + vector DB implementation
├── week3/ # Tool usage + memory system
├── week4/ # Multi-turn context agent
├── week5/ # Planning agents and loops
├── week6/ # Guardrails + evaluation
├── week7/ # Personalization + UI
├── week8/ # Final project showcase
└── README.md # This file
```

---

