import aiohttp
import asyncio
import openai
import logging
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = API_KEY

async def call_openai(messages: list[dict], model: str = "gpt-4.1-mini") -> str:
    messages.insert(1, {
        "role": "system",
        "content": "Use previous chat history only when it is relevant to the current question or task. If the current input is self-contained and does not require prior context, then ignore the previous history entirely. Prioritize the user’s latest message and avoid unnecessary references to earlier conversation unless they clearly add value."
    })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Error occurred while fetching answer from LLM: {str(e)}")
        return f"Error occurred while fetching answer from LLM: {str(e)}"