# INTRO-TO-AI-AGENT
SEASONS OF CODE, IIT BOMBAY

# Study Buddy - Week 1 Assignment

## 🧠 Objective
A simple command-line based assistant that answers questions based only on your study notes stored in `my_notes.md`.

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
Q: "When was the term Artificial Intelligence coined?"

A: "The term was coined in 1956..."

### 🟡 Partially Answered
Q: "Who are the top 5 most cited AI researchers?"

A: "I cannot find the answer in the provided notes." (Correct behavior if not mentioned)

### ❌ Not Answered in Notes
Q: "What is quantum computing?"

A: "I cannot find the answer in the provided notes."

## 📂 Files
**1. study_buddy_week1.py** - Main script.

**2. my_notes.md** - Your study notes.

**3. .env** - Contains your OpenAI API key (not committed to GitHub).

**4. README.md** - This file.


---

Let me know if you'd like help writing `my_notes.md` or setting up the GitHub repo.

