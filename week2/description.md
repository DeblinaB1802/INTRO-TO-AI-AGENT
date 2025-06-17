# INTRO-TO-AI-AGENT
SEASONS OF CODE, IIT BOMBAY

# 📚 Study Buddy - Week 2 Assignment

Continuing from Week 1, this assignment expands your Study Buddy into a smarter, Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on PDF notes using vector search and LLMs.

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

Place at least **3 PDF files** into a folder named `My_Notes/`.

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

Enter your question (or type 'exit' to quit):What is reinforcement learning?         

Study Buddy: Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on its actions, aiming to maximize cumulative rewards over time. RL is modeled as a Markov Decision Process, where the agent learns a policy to map states to actions. Applications include game playing, robotics, and autonomous driving. Algorithms like Q-learning and deep reinforcement learning, which combine RL with neural networks, have achieved remarkable success, such as AlphaGo defeating human champions. RL is particularly suited for sequential decision-making problems but can be computationally intensive and sensitive to reward design.

LLM response time : 2.65 seconds

____RETRIEVED DOCUMENT DETAILS____

     Retrieved Document 1:
Source : Machine Learning.pdf
Page : 4
Document chunk size : 2758
Cosine similarity : 0.5547
Content : 2 Types of Machine Learning
Machine learning is divided into three primary paradigms: supervised learning,
unsupervised learning, and reinforcement learning. Each type addresses dis-
tinct problems an

     Retrieved Document 2:
Source : Computer Vision.pdf
Page : 5
Document chunk size : 3002
Cosine similarity : 0.4024
Content : where xis the input image and ˆxis the reconstructed output. Unsupervised learning is valuable
for pretraining models when labeled data is scarce, enabling transfer learning in CV tasks.
3.3 Semi-Supe
```

##  🧪 Test Questions and Outputs

### Q1. ✅ Clearly Answered

**Question**: What are the types of machine learning?

**Answer**: Supervised, Unsupervised, and Reinforcement Learning are the three main types.

**Demonstrates**: Accurate chunk retrieval and relevant answer generation.

### Q2. 🟡 Partially Answered

**Question**: What is the difference between classification and clustering?

**Answer**: Clustering is an unsupervised learning technique that groups similar data points together based on their features. It does not use labeled data and is used to discover hidden patterns or structures in the data.

Classification is a supervised learning method where the model learns from labeled data to predict the category of new observations. It assigns inputs to predefined classes such as spam/not spam or disease/no disease.

**Demonstrates**: Some overlap found, but no complete match — answer is limited.

### Q3. ❌ Not Covered in Notes

**Question**: What is Quantization?

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
├── My_Notes/                  # Folder containing multiple PDF files
└── README.md                  # This file
```

---



