import requests
import openai

API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = "your_api_key"

openai.api_key = API_KEY
def call_openai(messages: list[dict], model: str = "gpt-4.1-mini") -> str:
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"}

    payload = {
        "model": model,
        "messages": messages
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    
    return f"API error. Status code : {response.status_code}"