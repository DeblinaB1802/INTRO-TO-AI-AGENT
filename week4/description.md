
# 🧠 Study Buddy Week 4 — An Interactive RAG + AI Agent

**Week 4 Study Buddy** is a Retrieval-Augmented Generation (RAG)-based AI assistant designed to help students engage with their study material intelligently. Instead of generating answers from scratch, it smartly searches your notes and delivers relevant, context-rich answers based on what *you’ve already studied*.

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
Input message:
```
Enter your question (or type 'exit' to quit):
```
#### 3. Ethical Check (Guardrail)
 - `check_ethical_compliance(question, query=True)`
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

####  Query Type: `math`
 - `plan_execute_refine_math()` → Handles multi-step math reasoning
 - No RAG or external search involved

####  Query Type: `wikipedia`
 - `search_wikipedia(structured_query)` → Retrieves Wikipedia summary
 - Prompt LLM with the wiki content
 - call_openai() → Generates response

####  Query Type: `tavily`
 - `search_tavily(structured_query)` → Uses Tavily API or similar external tool
 - Prompt LLM with the Tavily response
 - `call_openai()` → Generates response

####  Query Type: Other (Fallback)
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
## Example Session
```
Welcome to Study Buddy — your partner in learning. Let’s explore knowledge together!


Just a moment! Your Study Buddy is getting everything ready.


All set! Your Study Buddy is now ready to assist you.

Enter your educational details: (choose from ['middle_school', 'high_school', 'college', 'postgraduate', 'expert'] :college

Enter your question (or type 'exit' to quit):Explain me the concepts of Machine Learning.
rag
rag failed
***ANSWER***
Certainly! Let’s break down the core concepts of **machine learning (ML)** step-by-step, covering its main types, key components, and common real-world applications.

---

### 1. What is Machine Learning?

Machine learning is a branch of artificial intelligence (AI) focused on developing algorithms that enable computers to learn from and make decisions or predictions based on data, **without being explicitly programmed** for every specific task. Instead of hard-coding rules, ML models identify patterns and use them to generalize on new, unseen data.

---

### 2. Main Types of Machine Learning

There are three fundamental types of machine learning based on how the learning process is structured:

#### A. Supervised Learning
- **Definition**: The algorithm is trained on a **labeled dataset**, where each input data point is paired with the correct output (label).
- **Goal**: Learn a mapping from inputs (features) to outputs (labels) so it can predict labels for new inputs.
- **Examples**:
  - Classification (e.g., spam email detection, image recognition)
  - Regression (e.g., predicting house prices based on features)
- **Key aspect**: Requires labeled data for training.

#### B. Unsupervised Learning
- **Definition**: The algorithm deals with **unlabeled data** and tries to find hidden structures or patterns.
- **Goal**: Discover data groupings or underlying distributions without provided labels.
- **Examples**:
  - Clustering (e.g., customer segmentation)
  - Dimensionality reduction (e.g., principal component analysis for visualization)
- **Key aspect**: No labels needed; focuses on pattern discovery.

#### C. Reinforcement Learning
- **Definition**: An agent learns to make a sequence of decisions by interacting with an environment, receiving **rewards or penalties** based on its actions.
- **Goal**: Learn a policy (strategy) to maximize cumulative reward over time.
- **Examples**:
  - Game playing (e.g., AlphaGo)
  - Robotics control
- **Key aspect**: Learning through trial-and-error with feedback in form of rewards.

---

### 3. Key Components of Machine Learning

To build and understand ML systems, these components are essential:

#### A. Data
- Raw information used for learning.
- Can be structured (tables) or unstructured (images, text).
- Quality and quantity of data heavily influence model performance.

#### B. Features
- Measurable properties or attributes extracted from the data.
- Also called variables or predictors.
- Feature engineering (creating and selecting informative features) can critically improve model accuracy.

#### C. Models
- Mathematical representations or algorithms that learn patterns from data.
- Examples include decision trees, linear regression, neural networks.
- Model complexity varies from simple linear models to deep neural networks.

#### D. Training
- The process of feeding data to the model and adjusting its internal parameters (weights) to minimize error.
- Typically involves optimization techniques like gradient descent.
- Uses training datasets.

#### E. Evaluation
- Assessing model performance using metrics (accuracy, precision, recall, F1 score for classification; RMSE, MAE for regression).
- Involves testing on **unseen data** (test or validation sets) to check generalization.
- Important to avoid overfitting (when a model performs well on training data but poorly on new data).

---

### 4. Typical Real-World Applications of Machine Learning

- **Healthcare**: Disease diagnosis, personalized treatment recommendations, medical image analysis.
- **Finance**: Fraud detection, credit scoring, algorithmic trading.
- **Retail and E-commerce**: Recommendation systems (e.g., Amazon suggestions), customer segmentation, inventory demand forecasting.
- **Transportation**: Autonomous vehicles, traffic prediction.
- **Natural Language Processing**: Language translation, sentiment analysis, chatbots.
- **Computer Vision**: Facial recognition, object detection, image classification.
- **Robotics and Automation**: Industrial robot control, process optimization.

---

### Summary Table

| Concept            | Description                                  | Examples                         |
|--------------------|----------------------------------------------|---------------------------------|
| Supervised Learning | Learn from labeled data to predict outputs  | Spam detection, price prediction |
| Unsupervised Learning | Find patterns in unlabeled data             | Customer clustering, anomaly detection |
| Reinforcement Learning | Learn via rewards/penalties through interaction | Game playing, robotic control   |
| Data               | Raw information for learning                  | Text, images, sensor readings   |
| Features           | Input variables extracted from data           | Age, income, pixel values       |
| Models             | Algorithms that learn patterns                 | Neural networks, decision trees |
| Training           | Optimizing the model parameters                | Gradient descent optimization   |
| Evaluation         | Measuring model performance                     | Accuracy, F1 score, RMSE        |

---

If you want, I can also walk you through specific examples or dive deeper into any part, such as feature engineering or evaluation metrics. Just let me know!

Enter your question (or type 'exit' to quit):Teach me about Transformer in depth.
rag
***ANSWER***
The Transformer model represents a foundational advancement in machine learning, particularly in natural language processing (NLP). Introduced by Vaswani et al. in the 2017 paper "Attention is All You Need," the Transformer architecture revolutionized sequence processing by overcoming the limitations of prior models such as recurrent neural networks (RNNs). Below is an in-depth explanation covering the architecture, key components, training methodologies, and applications.

---

### 1. Transformer Architecture Overview

The Transformer architecture centers on the self-attention mechanism, which enables the model to weigh the relevance of different tokens in an input sequence relative to one another, capturing complex dependencies irrespective of token position. Unlike RNNs that process tokens sequentially and struggle with long-range relationships, Transformers process entire sequences simultaneously—leveraging parallel computation, leading to improved efficiency and scalability.

#### Core Elements:

- **Encoder-Decoder Structure:** The original Transformer is composed of two main parts:
  - **Encoder:** A stack of identical layers (usually six layers in the base model), each consisting of:
    - Multi-head self-attention sub-layer
    - Feed-forward neural network sub-layer
    - Residual connections and layer normalization following each sub-layer to stabilize training.
  - **Decoder:** Also a stack of identical layers that include:
    - Masked multi-head self-attention (to prevent attending to future tokens during training)
    - Cross-attention sub-layer to attend to the encoder's output
    - Feed-forward network and normalization layers.

- **Token Embeddings and Positional Encodings:**
  Since Transformers process input tokens in parallel and have no intrinsic notion of token order, positional encodings (typically sinusoidal functions or learned embeddings) are added to token embeddings to inject sequence order information.

---

### 2. Key Components

#### A. Self-Attention Mechanism

- The self-attention mechanism computes a weighted representation of each token by attending to every other token in the sequence. Intuitively, this allows each token’s representation to be informed by the entire context, capturing syntactic and semantic relationships.
- **Multi-head Self-Attention:** Several attention "heads" run in parallel, each learning different aspects of the dependencies and patterns, enhancing the model’s ability to capture varied linguistic features.

#### B. Feed-Forward Neural Networks

- After attention, each token embedding passes through a position-wise feed-forward network (typically a two-layer fully connected network with activation functions), which provides non-linear transformations and increases model expressiveness.

#### C. Residual Connections and Layer Normalization

- To facilitate stable and efficient training, residual (skip) connections add the input of each sub-layer back to its output, while layer normalization standardizes activations, improving gradient flow and convergence.

---

### 3. Variants of Transformer Configurations in NLP

- **Encoder-Only Models:** Examples include BERT, which uses the encoder stack to generate contextualized embeddings. This configuration is optimized for understanding tasks like classification, question answering, and named entity recognition.
- **Decoder-Only Models:** GPT models are decoder stacks trained autoregressively to generate text, predicting the next token based on previous tokens.
- **Encoder-Decoder Models:** T5 is an example that combines both, enabling versatile tasks such as machine translation, summarization, or any sequence-to-sequence mapping where understanding and generation are both needed.

---

### 4. Training Process

#### A. Pre-Training

- Transformers, especially large language models (LLMs), undergo pre-training on massive corpora of text to capture general linguistic patterns.
- Common objectives:
  - **Masked Language Modeling (MLM):** Random tokens in the input are masked, and the model learns to predict the masked tokens from context (used in BERT).
  - **Causal Language Modeling (CLM):** The model predicts the next token in a sequence given all preceding tokens (used in GPT).
- This stage equips the model with a broad, general understanding of language.

#### B. Fine-Tuning

- After pre-training, the Transformer model is fine-tuned on smaller, task-specific datasets to specialize in particular applications.
- Fine-tuning involves supervised learning, updating model weights to optimize performance on the target task while leveraging learned generalizations.
- Techniques include full fine-tuning (all parameters updated) and parameter-efficient fine-tuning (updating only adapters or low-rank modules to save computational resources and preserve original knowledge).

---

### 5. Applications in NLP and Beyond

- **Natural Language Processing:**
  - Text generation (e.g., GPT models generating human-like text)
  - Machine translation (via encoder-decoder models)
  - Sentiment analysis, question answering, and classification (encoder-based models)
  - Summarization and text completion
- **Other Domains:**
  - While originally purposed for NLP, Transformer-based architectures have extended into vision (Vision Transformers), speech processing, and multimodal tasks (integrating text, images, audio).       
  - Emerging research explores sparse Transformers and efficient attention mechanisms to handle very long sequences or large-scale data more effectively.

---

### Summary

The Transformer model transforms sequence modeling by employing self-attention mechanisms that capture contextual relations across entire sequences simultaneously. Its architecture balances multi-head attention, feed-forward layers, and normalization strategies to enable deep, stable learning. Through large-scale pre-training and targeted fine-tuning, Transformers excel in diverse NLP tasks and continue to push boundaries in other fields due to their flexibility, parallelizability, and strong representational capabilities. Advanced variants and ongoing research into efficiency and fairness promise to further extend the Model’s reach and impact.

Enter your question (or type 'exit' to quit): Help me solve this numerical problem. 2x^2 + 4x + 2 = 0
math
***ANSWER***
 Plan:
1. Write down the quadratic equation: \(2x^2 + 4x + 2 = 0\).

2. Identify the coefficients \(a\), \(b\), and \(c\) from the equation, where \(a = 2\), \(b = 4\), and \(c = 2\).

3. Calculate the discriminant \(\Delta = b^2 - 4ac\).

4. Determine the nature of the roots based on the discriminant (\(\Delta > 0\), \(\Delta = 0\), or \(\Delta < 0\)).

5. Use the quadratic formula \(x = \frac{-b \pm \sqrt{\Delta}}{2a}\) to find the roots.

6. Simplify the expressions under the square root and then compute the values of \(x\).

7. Write down the final roots as the solution to the equation.

 Solution:
The given solution is mathematically correct, logically sound, and clearly presented. However, the formatting and flow can be improved slightly for better readability and completeness. Here is the polished and clearly formatted version:

---

**Solution:**

Let's solve the quadratic equation step by step.

**Given equation:**
\[
2x^2 + 4x + 2 = 0
\]

---

**Step 1: Identify the coefficients**
\[
a = 2, \quad b = 4, \quad c = 2
\]

---

**Step 2: Calculate the discriminant**
\[
\Delta = b^2 - 4ac = 4^2 - 4 \times 2 \times 2 = 16 - 16 = 0
\]

---

**Step 3: Determine the nature of the roots**
Since \(\Delta = 0\), the quadratic equation has exactly one real root (a repeated root).

---

**Step 4: Apply the quadratic formula**
\[
x = \frac{-b \pm \sqrt{\Delta}}{2a}
\]

Substitute the values:
\[
x = \frac{-4 \pm \sqrt{0}}{2 \times 2} = \frac{-4 \pm 0}{4} = \frac{-4}{4} = -1
\]

---

**Step 5: Write the final answer**
The quadratic equation has one real root with multiplicity 2:
\[
\boxed{x = -1}
\]

---

**Summary:**
The quadratic equation \(2x^2 + 4x + 2 = 0\) has a single repeated root, \(x = -1\).

---

If you want, I can also show how to factor the quadratic using this root. Would you like that?

Enter your question (or type 'exit' to quit):Tell me about the recent development in AI?
tavily
***ANSWER***
As of mid-2025, artificial intelligence has witnessed several significant advancements and breakthroughs across key technologies, research domains, and practical applications. Below is a detailed summary based on the latest reports and news:

1. **Cutting-Edge AI Models and Capabilities**

   - **Google’s Gemini 2.0:** Google released Gemini 2.0, described as its most capable AI model to date. This model incorporates *agentic capabilities*, which refer to the AI’s ability to autonomously perform complex tasks, make decisions, and interact dynamically with developers, enterprises, and end-users. Gemini 2.0 exemplifies a major step forward in versatile AI agents that can adapt across multiple domains, enhancing productivity and creativity.

   - **Advanced Image and Video Generation:** Google also introduced next-generation generative AI models for multimedia content creation:
     - *Veo 2*: An AI model specialized in video generation, capable of creating high-quality video content with advanced realism and coherence.
     - *Imagen 3*: An improved image generation model achieving state-of-the-art results in direct comparisons with other leading models. These advancements enable more sophisticated and accessible multimedia production, pushing creative boundaries in entertainment, marketing, and design sectors.

2. **AI in Environmental Conservation**

   - **Microsoft’s AI for Endangered Giraffes:** Microsoft’s AI tools are actively employed in wildlife conservation, particularly for endangered giraffe species in Africa. By analyzing drone footage and camera trap data, the AI system identifies migration patterns and potential poaching threats with greater speed and accuracy than traditional methods. This application falls under Microsoft’s broader *AI for Earth* initiative that leverages AI to address environmental challenges, showcasing how AI enhances ecological monitoring and protection efforts.

3. **Brain-Computer Interface Transformations**

   - **Australian Mind-Reading AI:** Researchers in Australia have developed a pioneering brain-computer interface (BCI) that interprets neural signals associated with imagined speech and translates them into textual words. This AI-enabled system achieves over 70% accuracy in decoding thoughts into readable text, representing a groundbreaking fusion of neuroscience and AI. This technology holds transformative potential for individuals with speech or motor disabilities, offering a new mode of communication and interaction that bypasses traditional physical channels.

4. **Broader Industry and Research Trends**

   - According to Stanford University’s 2025 AI Index Report, the AI field continues rapid growth and diversification:
     - Research is focused on improving *generalization* and *transfer learning* capabilities, enabling AI models to apply learned knowledge across diverse tasks.
     - Industry adoption expands across sectors including healthcare, finance, creative industries, and environmental sustainability.
     - Governance and ethical frameworks are evolving to address AI’s societal impacts, including fairness, transparency, and accountability.

### Summary

In summary, by mid-2025, AI advancements are characterized by:

- Development of advanced, agentic AI models like Google’s Gemini 2.0 with autonomous decision-making abilities.
- Cutting-edge generative models for high-fidelity video and image synthesis (Veo 2 and Imagen 3).
- Practical application of AI for ecological conservation, exemplified by Microsoft’s use in giraffe protection.
- Revolutionary brain-computer interfaces translating thoughts into speech, enhancing accessibility.
- Broad research and industry trends prioritizing adaptability, ethical governance, and cross-domain deployment.

These breakthroughs collectively demonstrate AI’s expanding capabilities, scope of impact, and integration into diverse aspects of science, technology, and society.

Enter your question (or type 'exit' to quit):can you help me give a road map to DSA?
rag
rag failed
***ANSWER***
Absolutely! Mastering Data Structures and Algorithms (DSA) is a great goal that will strengthen your problem-solving skills and prepare you for coding interviews, competitive programming, and efficient software development. Below is a **detailed and structured roadmap** organized into phases, with key topics, suggested learning resources, and practical steps to follow.

---

## Roadmap for Mastering Data Structures and Algorithms (DSA)

### Phase 1: Foundations and Basic Concepts

**Goal**: Understand the basics of Algorithms and Data Structures, complexity analysis, and simple problem solving.

#### Key Topics:
- **Algorithm Basics:** Definition, real-life examples
- **Mathematical Background:** Big-O notation, Time & Space complexity, best/average/worst cases
- **Basic Data Structures:** Arrays, Strings, Linked Lists (Singly, Doubly)
- **Basic Algorithms:** Simple sorting (Bubble, Selection, Insertion), Searching (Linear, Binary Search)
- **Recursion:** Base case, recursive calls, stack behavior

#### Learning Resources:
- Book: *“Grokking Algorithms”* by Aditya Bhargava (Visual and beginner-friendly)
- Online Course: Coursera - *“Algorithms, Part I”* by Princeton University
- Website: GeeksforGeeks (Basics sections)

#### Practical Steps:
- Implement all basic data structures from scratch in your preferred programming language.
- Solve easy-level problems on arrays, strings, and linked lists on sites like LeetCode, HackerRank.
- Practice writing recursive functions (e.g., factorial, Fibonacci, binary search).

---

### Phase 2: Core Data Structures and Fundamental Algorithms

**Goal**: Master key data structures and fundamental algorithms systematically.

#### Key Topics:
- Data Structures:
  - Stacks and Queues (including deque)
  - Trees (Binary Trees, Binary Search Trees, Tree traversal methods)
  - Heaps (Min-heap, Max-heap)
  - Hash Tables / Hash Maps
- Algorithms:
  - Sorting (Merge sort, Quick sort, Heap sort)
  - Searching (Binary search in sorted arrays and trees)
  - Basic Graph theory (representation: adjacency list & matrix)
- Algorithm Paradigms:
  - Divide and Conquer
  - Greedy Algorithms
  - Intro to Dynamic Programming

#### Learning Resources:
- Book: *“Introduction to Algorithms”* (CLRS) by Cormen et al. — start with chapters relevant to above topics
- MIT OpenCourseWare - *Introduction to Algorithms (6.006)*
- YouTube Channels: Abdul Bari, Tushar Roy (DS & Algo playlists)

#### Practical Steps:
- Code each data structure and algorithm from scratch.
- Solve medium problems emphasizing these data structures.
- Analyze problems to decide which data structure or paradigm fits best.

---

### Phase 3: Advanced Topics and Algorithmic Techniques

**Goal**: Learn complex data structures and advanced algorithmic paradigms for deeper understanding and optimization.

#### Key Topics:
- Advanced Trees: AVL, Red-Black Trees, Segment Trees, Fenwick Trees (Binary Indexed Trees)
- Graph algorithms:
  - BFS, DFS, Topological Sort
  - Shortest Paths (Dijkstra, Bellman-Ford, Floyd-Warshall)
  - Minimum Spanning Tree (Prim’s, Kruskal’s)
  - Network Flow basics
- Dynamic Programming (DP):
  - Memoization, Tabulation
  - Classic DP problems (Knapsack, Longest Common Subsequence, Matrix Chain Multiplication)
- Backtracking and Branch & Bound
- String algorithms:
  - Trie
  - KMP Algorithm, Rabin-Karp
- Bit Manipulation Techniques

#### Learning Resources:
- Book: *“Algorithms”* by Robert Sedgewick and Kevin Wayne (Princeton)
- Stanford Graph Theory Lectures (YouTube)
- TopCoder tutorials on advanced topics
- HackerRank and Codeforces tutorials for challenges on DP, Graphs, and Trees

#### Practical Steps:
- Implement complex data structures and test thoroughly.
- Solve classic algorithmic problems tagged with these topics.
- Participate in coding contests to apply algorithms under time constraints.
- Explore optimization techniques and problem-solving tricks.

---

### Phase 4: Mastery and Application

**Goal**: Achieve fluency in problem-solving, optimize solutions, and prepare for coding interviews or competitive programming.

#### Key Topics:
- Problem-solving patterns and strategies
- Optimization techniques (time, space)
- Parallel and Distributed Algorithms (introductory)
- Real-world applications of DSA
- System design basics related to data structures

#### Learning Resources:
- Books: *“Competitive Programming”* by Steven Halim
- Websites: LeetCode (Explore section), Codeforces, AtCoder contests
- Interview preparation platforms: InterviewBit, Pramp (mock interviews)
- Books: *“Cracking the Coding Interview”* by Gayle Laakmann McDowell

#### Practical Steps:
- Solve daily challenges at varying difficulty levels.
- Time yourself to improve speed and accuracy.
- Review and optimize previous solutions.
- Regularly participate in live contests.
- Conduct mock interviews and discuss solutions with peers or mentors.

---

## Additional Tips for Effective Learning

1. **Consistent Practice:** DSA concepts improve with hands-on coding. Aim for daily or regular practice.
2. **Understand Problem Patterns:** Categorize problems by underlying concepts rather than memorizing solutions.
3. **Debugging Skills:** Learn to debug efficiently to understand problems deeply.
4. **Peer Learning:** Join study groups or forums to discuss challenging problems.
5. **Use Visual Aids:** Tools like VisuAlgo or AlgoExpert explain how algorithms run step-by-step visually.
6. **Write and Review:** Comment your code and maintain notes summarizing each concept and important points.

---

If you want, I can help design a **weekly schedule** based on your available time or recommend specific problem sets tailored to your current level. Would you like that?

Enter your question (or type 'exit' to quit):Should I support racist people?
⚠️ **Caution**: Your query includes a potentially sensitive term *'racist'* related to *hate speech and bias*.
                    To maintain a respectful and inclusive learning environment, Study Buddy avoids promoting such language.
                    Please consider rephrasing. 😊


Enter your question (or type 'exit' to quit):I am getting bullied at college what should I do?
rag
rag failed
***ANSWER***
I'm really sorry to hear that you're experiencing this. Bullying can be very distressing, but there are practical steps and resources you can use to protect yourself and address the situation. Here’s a structured approach to help you navigate this:

### 1. **Recognize and Document the Bullying**
- **Identify the behavior:** Bullying can include verbal abuse, social exclusion, online harassment, or physical intimidation.
- **Keep a record:** Maintain a detailed log of incidents, including dates, times, locations, what was said or done, and any witnesses. This documentation is useful if you report the bullying.

### 2. **Ensure Your Safety**
- Avoid situations where the bullying occurs if possible.
- Stay in groups or public areas when you feel vulnerable.
- Inform trusted friends or classmates about what’s happening so they can support you and provide assistance if needed.

### 3. **Seek Support**
- **Talk to trusted people:** Share your experience with close friends, family members, or mentors. Emotional support is important.
- **Utilize campus resources:**
  - **Counseling services:** Most colleges have counseling or mental health services where you can speak confidentially with professionals.
  - **Student support offices:** Offices like student affairs, dean of students, or diversity and inclusion centers often have protocols for handling bullying and harassment.
  - **Campus security:** If you feel threatened, contact campus security for immediate help.

### 4. **Report the Bullying**
- Approach the designated authority at your college (e.g., student conduct office, campus safety, or human resources).
- Submit your documentation and request a formal investigation.
- Be aware of your college’s anti-bullying and harassment policies—they usually outline procedures and protections.

### 5. **Consider Professional Help**
- If bullying severely impacts your mental health, consider speaking to a mental health professional outside campus.
- They can provide coping strategies and support.

### 6. **Legal Protections**
- Depending on your location, certain types of bullying may be illegal, especially if related to discrimination (race, gender, disability).
- If bullying continues and campus measures don’t work, you might consider legal advice.

---

### Summary Example:

**You notice a classmate repeatedly insults you and spreads rumors.**
- You **document** each incident.
- You **talk to a counselor** at your college for emotional support.
- You **inform student affairs** to report the bullying formally.
- You ensure you stay close to friends and avoid being alone in vulnerable situations.
- If the bullying escalates to threats, you contact campus security immediately.

---

If you’d like, I can help you draft a report or give advice on how to phrase your experience when talking to campus officials. Your safety and well-being matter a lot—taking these steps is a strong way to regain control of the situation.

Enter your question (or type 'exit' to quit):Tell me about Sita abduction?       
wiki
WARNING:root:No page found: Page id "Please provide a detailed explanation of the abduction of Sita as described in the Hindu epic Ramayana, including the circumstances, key characters involved, and its significance in the story." does not match any pages. Try another id!
self-correction
***ANSWER***
Certainly! Here is a detailed explanation of the abduction of Sita as described in the Hindu epic Ramayana:

**Circumstances:**
The abduction of Sita occurs during the 14-year exile of Rama, the prince of Ayodhya, his wife Sita, and his brother Lakshmana. They are living in the forest to fulfill a vow. One day, Sita sees a beautiful golden deer near their hermitage and is captivated by it. She asks Rama to capture the deer for her. Rama pursues the deer, which is actually the demon Maricha in disguise, but he eventually kills it after a long chase.

**Key Characters Involved:**
- **Sita:** Rama’s devoted wife, known for her purity and virtue.
- **Rama:** The hero of the epic, an incarnation of the god Vishnu, devoted to dharma (righteousness).
- **Lakshmana:** Rama’s loyal younger brother who guards Sita.
- **Maricha:** A demon who transforms into the golden deer to lure Rama and Lakshmana away.
- **Ravana:** The demon king of Lanka, who abducts Sita.

**How the Abduction Happens:**
After Maricha transforms into the golden deer and is chased by Rama, Sita insists that Rama must capture the deer alive for her. Rama pursues Maricha deep into the forest and eventually shoots him with an arrow. Before dying, Maricha mimics Rama’s voice, crying out for help.

Hearing this, Sita becomes frightened and urges Lakshmana to go help Rama. Despite his doubts, Lakshmana reluctantly leaves, but before leaving, he draws a protective boundary known as the “Lakshmana Rekha” around their hut, instructing Sita not to cross it to stay safe.

With both Rama and Lakshmana away, Ravana arrives in disguise as a mendicant (holy hermit) seeking alms. Bound by her respect for guests and dharma, Sita steps outside the protective boundary to give him food. Ravana then reveals his true form, kidnaps Sita, and flies away on his flying chariot, the Pushpaka Vimana, taking her to his kingdom in Lanka.

**Significance in the Ramayana:**
Sita’s abduction is the central turning point of the Ramayana. It sets the stage for Rama’s heroic quest to rescue her, which involves forging alliances (notably with the monkey king Sugriva and the devoted Hanuman), building a massive army, and waging a great war against Ravana and his forces in Lanka.

This event highlights several key themes of the epic, including:
- The struggle between good and evil.
- The importance of dharma (righteous duty).
- The virtues of loyalty, devotion, and courage.
- The testing of Sita’s chastity and Rama’s role as a righteous leader.

Ultimately, the rescue of Sita and the defeat of Ravana restore cosmic order and dharma, reinforcing the moral and spiritual lessons of the Ramayana.

If you want, I can share more about the events that follow the abduction or other aspects of the Ramayana!

Enter your question (or type 'exit' to quit):exit
Good Bye!!!
```
## 📹 Project Demo Video Link
[[https://drive.google.com/file/d/1xBNhwK9bmEslJY_1L3uOBcaSaHBXIbtk/view?usp=drive_link](https://drive.google.com/file/d/1xBNhwK9bmEslJY_1L3uOBcaSaHBXIbtk/view?usp=sharing)]


