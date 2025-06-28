import json
import os
import spacy
import tiktoken
from datetime import datetime
from call_llm import call_openai
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

CHAT_HISTORY_FILE = r"week4\conversation_history.json" 
chat_history = []
entities = {}

def load_chat_history():
    """Load conversation history from file."""
    global chat_history
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            chat_history = json.load(f)

def save_history():
    """Save conversation history to file."""
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(chat_history, f, indent=2)

def add_to_history(role: str, content: str):
    "Save current conversation to the conversation history."
    chat_history.append({
        "role" : role,
        "content" : content,
        "timestamp" : datetime.now().isoformat()
    })
    save_history()

def count_token(text: str, model_name: str = "gpt-4") -> int:
    """Estimate the number of tokens in a given string"""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def get_relevant_history(query: str, max: int = 2) -> list[str]:
    """Retrieve relevant history based on keyword matching."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    query_words = word_tokenize(query)
    filtered_query_words = [stemmer.stem(word) for word in query_words if word.isalpha() and word not in stop_words]
    
    relevant_msg = []
    for chat in chat_history:
        msg_words = word_tokenize(chat.get("content", " "))
        filtered_msg_words = [stemmer.stem(word) for word in msg_words if word.isalpha() and word not in stop_words]
        overlap = len(set(filtered_query_words) & set(filtered_msg_words))
        if overlap > 0:
            relevant_msg.append((overlap, chat))
    ranked_msg = sorted(relevant_msg, key=lambda x: x[0], reverse=True)
    if len(ranked_msg) >= max:
            top_ranked_msg = [chat for _, chat in ranked_msg[:max]]
    else:
        top_ranked_msg = [chat for _, chat in ranked_msg]
    return top_ranked_msg

def get_optimized_context(query: str, max_tokens: int) -> list[dict]:
    """Combine recent and relevant history."""
    if len(chat_history) > 2:
        recent_msgs = chat_history[-2:]
    else:
        recent_msgs = chat_history[:]
    relevant_msgs = get_relevant_history(query, max = 2)
    final_msgs = list({msg["content"]: msg for msg in recent_msgs + relevant_msgs}.values())
    total_tokens = 0
    final_history = []
    for msg in final_msgs:
        msg_tokens = count_token(msg['content'])
        if total_tokens + msg_tokens < max_tokens:
            final_history.append(msg)
            total_tokens += msg_tokens
        else:
            break
    return final_history

def extract_entities(text: str) -> dict:
    """Extract and store words as entities."""
    text_doc = nlp(text)
    for entity in text_doc.ents:
        entities[entity] = {"label" : {entity.label_},
                            "content" : text,
                            "timestamp" : datetime.now().isoformat()
                            }

def get_entity_context(entity: str) -> str:
    """Retrieve entity context if mentioned."""
    return entities.get(entity, {}).get("context", "")

def summarize_history() -> str:
    """Summarize conversation history using OpenAI."""
    load_chat_history()
    full_context = " ".join([chat["content"] for chat in chat_history])
    if count_token(full_context) < 50:
        return full_context
    
    summary_prompt = f"Summarize the following conversation concisely:\n{full_context}"
    summary_context = [{"role" : "user", "content" : summary_prompt}]
    summary = call_openai(summary_context)
    return summary
