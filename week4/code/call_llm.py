import requests
import openai
import logging
import os

API_URL = os.getenv("OPENAI_API_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = API_KEY
def call_openai(messages: list[dict], model: str = "gpt-4.1-mini") -> str:

    messages.insert(1, {"role": "system", "content": "Use previous chat history only when it is relevant to the current question or task. If the current input is self-contained and does not require prior context, then ignore the previous history entirely. Prioritize the user’s latest message and avoid unnecessary references to earlier conversation unless they clearly add value."})
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"}

    payload = {
        "model": model,
        "messages": messages
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()["choices"][0]["message"]["content"]
        return result
        
    except Exception as e:
        logging.error(f"Error occurred while fetching answer from LLM: {str(e)}")
        return f"Error occurred while fetching answer from LLM: {str(e)}"