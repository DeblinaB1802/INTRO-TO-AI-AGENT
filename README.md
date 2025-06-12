# INTRO-TO-AI-AGENT
SEASONS OF CODE, IIT BOMBAY

# Study Buddy - Week 1 Assignment

## 🧠 Objective
The goal of this project is to build a simple question-answering assistant, kind of like a mini AI study buddy. You give it a Markdown file `my_notes.md` with your study notes on any academic topic, and it can answer questions based only on what's written in those notes.

This tool is helpful when you're studying and want quick answers without searching through your notes manually. If the answer to your question isn’t in the notes, the assistant will honestly say, “I cannot find the answer in the provided notes.” That way, you know exactly what your notes cover and what they don’t.

In short, this project helps you:

- Practice working with language models (like OpenAI’s GPT).

- Learn how to extract answers from documents using AI.

- Build a helpful tool that actually supports your own learning process.

It's simple, but powerful — and it’s just the beginning of building smarter, personalized study tools with AI.

## 🛠️ Setup Instructions

1. Clone this repository.
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies:

   ```bash
   pip install langchain langchain-openai python-dotenv

4. Set your OpenAI API key in a .env file:

   ```ini
   OPENAI_API_KEY=your_openai_key_here

5. Run the assistant:

   ```bash
   python study_buddy_week1.py

## 🧪 Testing
We tested the assistant with three types of questions:

### ✅ Clearly Answered
Q: What is the need for CNN?

A: The need for Convolutional Neural Networks (CNNs) arises from the inefficiency and computational complexity of traditional neural networks when dealing with high-dimensional data like images. Images consist of a large number of pixels, leading to overfitting, slow training, and high memory usage in traditional neural networks. CNNs address this challenge by introducing a more efficient architecture that leverages spatial hierarchies in image data, reducing the number of parameters and allowing the network to generalize better.

### 🟡 Partially Answered
Q: What is residual connections?

A: Residual connections are a concept introduced in the ResNet architecture, revolutionizing CNN design. They allow very deep networks to be trained without degradation in performance by creating shortcuts, or skip connections, that bypass one or more layers. This helps in mitigating the vanishing gradient problem and allows for easier training of deep neural networks.

### ❌ Not Answered in Notes
Q: What is NLP in AI?

A: I can't find the answer in the provided notes.

## 📂 Files
**1. `study_buddy_week1.py`** - Main script.

**2. `my_notes.md`** - Your study notes.

**3. `.env`** - Contains your OpenAI API key (not committed to GitHub).

**4. `README.md`** - This file.


# 📚 Study Buddy - Week 2 Assignment

Continuing from Week 1, this assignment expands your Study Buddy into a smarter, Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on both Markdown and PDF notes using vector search and LLMs.

---

## 🧠 Objective

In **Week 2**, the Study Buddy evolves into a RAG-based assistant capable of:

* Reading multiple **PDF notes** instead of just Markdown.
* Breaking down documents into smaller chunks.
* Generating **semantic embeddings** using `sentence-transformers`.
* Creating a **vector database** using **ChromaDB**.
* Retrieving the most relevant chunks for a given query.
* Responding using **OpenAI's GPT**, strictly based on retrieved chunks.
* Clearly stating **“I can’t find the answer in the provided notes”** if nothing relevant is found.

Additionally, the assistant provides **document traceability** for answers — including the **file name**, **page number**, and **similarity score**.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your_repo_url>
cd week2/
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set OpenAI API Key

Create a `.env` file in the root directory and add:

```env
OPENAI_API_KEY=your_openai_key_here
```

### 5. Add Your Study Notes

Place at least **3 PDF files** into a folder named `notes/` under `week2/`.

---

## ▶️ Run the Assistant

```bash
python study_buddy_week2.py
```

---

## 💬 Example Session

```
Welcome to Study Buddy — your partner in learning. Let’s explore knowledge together!

Just a moment! Your Study Buddy is getting everything ready.
  >> Breaking your notes into smaller, manageable pieces...
  >> Creating embeddings to understand your notes better...
  >> Setting up a smart search system with your notes...
Model successfully initialized...

All set! Your Study Buddy is now ready to assist you.
```

---

## ✅ Test Questions and Outputs

### Q1. ❓ Clearly Answered

**Question**: What are the types of machine learning?
**Answer**: Supervised, Unsupervised, and Reinforcement Learning are the three main types...
**Demonstrates**: Accurate chunk retrieval and relevant answer generation.

### Q2. 🟡 Partially Answered

**Question**: What is the difference between classification and clustering?
**Answer**: The notes mention both but do not explain the difference clearly.
**Demonstrates**: Some overlap found, but no complete match — answer is limited.

### Q3. ❌ Not Covered in Notes

**Question**: What is BERT?
**Answer**: I can’t find the answer in the provided notes.
**Demonstrates**: The assistant respects scope and avoids hallucination.

---

## 📂 Project Structure

```
week2/
│
├── study_buddy_week2.py       # Main RAG assistant script
├── requirements.txt           # Python dependencies
├── .env                       # API key (excluded from version control)
├── notes/                     # Folder containing multiple PDF files
└── README.md                  # This file
```

---

## 🚀 Assignment 2.2: Alternate Vector Store (ChromaDB)

In this implementation, we replaced FAISS with **ChromaDB**, a flexible, serverless vector database.

### ✅ ChromaDB Functionality:

* Stores vector representations of note chunks.
* Retrieves top-k similar chunks for a user query.
* Maintains **chunk-level metadata**: source file name, page number, chunk size, similarity score.
* Enhances transparency by showing where each answer comes from.

---

## 🧪 Challenges Faced

* **PDF Parsing Variability**: Some PDFs had poor formatting or lacked newline markers. Resolved using better chunking logic.
* **Noisy Chunks**: Very short or empty lines were filtered to avoid meaningless embeddings.
* **Debugging Retrieval**: Printing retrieved chunks helped verify that answers were grounded in actual content.

---

## 🧠 Learnings

* How to integrate document processing, vector search, and LLMs into a cohesive pipeline.
* Best practices in prompt design for retrieval-augmented generation.
* The importance of explainability when building AI tools for learning.

---

## 📌 Notes

* This version does not hallucinate — it only answers from available content.
* Retrieval traceability builds trust in the assistant’s answers.
* Easily extendable to work with `.txt` or `.md` files.

---

**Let’s keep building smarter tools together!**
🧑‍💻 *Made with 💡 by Deblina Biswas · CSRE, IIT Bombay*



---


