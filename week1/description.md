# INTRO-TO-AI-AGENT
SEASONS OF CODE, IIT BOMBAY

# 📚 Study Buddy - Week 1 Assignment

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

5. Add Your Study Notes

   Place at **1 Markdown file** name `my_notes.md` into a folder named `week1/`.

5. Run the assistant:

   ```bash
   python study_buddy_week1.py

## 💬 Example Session

```
Model successfully initialized...

Enter your question (or type 'exit' to quit):What is CNN?

Study Buddy: CNN stands for Convolutional Neural Network.

LLM response time : 1.34 seconds

Enter your question (or type 'exit' to quit):What do you understand by CNN?

Study Buddy: CNN stands for Convolutional Neural Networks. They are a powerful deep learning architecture that revolutionizes the way computers process visual data. CNNs are specifically designed to handle high-dimensional data like images by exploiting spatial hierarchies and local patterns in the data. They consist of convolutional layers, pooling layers, and fully connected layers, each with a specific role in processing visual information. CNNs are widely used in various applications such as image classification, object detection, facial recognition, medical diagnostics, autonomous vehicles, agriculture, retail, and more.

LLM response time : 2.13 seconds

Enter your question (or type 'exit' to quit):What are different CNN architectures?

Study Buddy: 1. LeNet-5
2. AlexNet
3. VGGNet
4. GoogLeNet (Inception)
5. ResNet

LLM response time : 1.30 seconds

Enter your question (or type 'exit' to quit):exit
```

## 🧪 Testing
We tested the assistant with three types of questions:

### Q1. ✅ Clearly Answered
**Question**: What is the need for CNN?

**Answer**: The need for Convolutional Neural Networks (CNNs) arises from the inefficiency and computational complexity of traditional neural networks when dealing with high-dimensional data like images. Images consist of a large number of pixels, leading to overfitting, slow training, and high memory usage in traditional neural networks. CNNs address this challenge by introducing a more efficient architecture that leverages spatial hierarchies in image data, reducing the number of parameters and allowing the network to generalize better.

**Demonstrates**: Accurate chunk retrieval and relevant answer generation.

### Q2. 🟡 Partially Answered
**Question**: What is residual connections?

**Answer**: Residual connections are a concept introduced in the ResNet architecture, revolutionizing CNN design. They allow very deep networks to be trained without degradation in performance by creating shortcuts, or skip connections, that bypass one or more layers. This helps in mitigating the vanishing gradient problem and allows for easier training of deep neural networks.

**Demonstrates**: Some overlap found, but no complete match — answer is limited.

### Q3. ❌ Not Answered in Notes
**Question**: What is NLP in AI?

**Answer**: I can't find the answer in the provided notes.

**Demonstrates**: The assistant respects scope and avoids hallucination.

## 📂 Project Structure

```
week1/
│
├── study_buddy_week1.py       # Main assistant script
├── .env                       # API key (excluded from version control)
├── my_notes.md                # Your study notes
└── README.md                  # This file
```

