import requests
import wikipedia
import logging
import os
from dotenv import load_dotenv

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

def search_tavily(query: str, api_key: str = tavily_api_key, timeout: int = 10) -> str:
    """
    Perform a search using the Tavily API and return a concise answer.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_sources" : True
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "TavilyQueryAgent/1.0"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        try:
            answer = data.get("answer")
            if len(answer) < 500 and data.get("results"):
                fallback_info = []
                for src in data["results"][:2]:  # top 2 sources
                    title = src.get("title", "")
                    content = src.get("content", "")
                    url = src.get("url", "")
                    fallback_info.append(f"From {title}:\n{content}\nMore: {url}\n")

                full_answer = answer + "\n\nAdditional Details:\n" + "\n".join(fallback_info)
            else:
                full_answer = answer
            return full_answer
        except:
            return "No answer found."

    except requests.exceptions.Timeout:
        logging.warning("Tavily API request timed out.")
        return "Search failed."

    except requests.exceptions.ConnectionError:
        logging.error("Failed to connect to Tavily API.")
        return "Search failed."

    except requests.exceptions.HTTPError as e:
        logging.error(f"Tavily API HTTP error: {e.response.status_code} - {e.response.text}")
        return "Search failed."

    except Exception as e:
        logging.exception(f"Unexpected error during Tavily search: {e}")
        return "Search failed."

def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a 10-sentence summary or an error message."""

    try:
        return wikipedia.summary(query, sentences=10)
    
    except wikipedia.exceptions.DisambiguationError as e:
        logging.warning(f"Multiple options found: {str(e)}")
        return "Search Failed."
    
    except wikipedia.exceptions.PageError as e:
        logging.warning(f"No page found: {str(e)}")
        return "Search Failed."
    
    except Exception as e:
        logging.error(f"Unexpected error occurred while fetching Wikipedia content: {str(e)}")
        return "Search Failed."