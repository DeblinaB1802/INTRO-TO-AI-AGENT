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
Q: "What is the need for CNN?"

A: "The need for Convolutional Neural Networks (CNNs) arises from the inefficiency and computational complexity of traditional neural networks when dealing with high-dimensional data like images. Images consist of a large number of pixels, leading to overfitting, slow training, and high memory usage in traditional neural networks. CNNs address this challenge by introducing a more efficient architecture that leverages spatial hierarchies in image data, reducing the number of parameters and allowing the network to generalize better."

### 🟡 Partially Answered
Q: "What is residual connections?"

A: "Residual connections are a concept introduced in the ResNet architecture, revolutionizing CNN design. They allow very deep networks to be trained without degradation in performance by creating shortcuts, or skip connections, that bypass one or more layers. This helps in mitigating the vanishing gradient problem and allows for easier training of deep neural networks."

### ❌ Not Answered in Notes
Q: "What is NLP in AI?"

A: "I can't find the answer in the provided notes."

## 📂 Files
**1. study_buddy_week1.py** - Main script.

**2. my_notes.md** - Your study notes.

**3. .env** - Contains your OpenAI API key (not committed to GitHub).

**4. README.md** - This file.


---

Let me know if you'd like help writing `my_notes.md` or setting up the GitHub repo.

