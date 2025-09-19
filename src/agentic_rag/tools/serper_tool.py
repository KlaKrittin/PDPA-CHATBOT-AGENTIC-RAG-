import os
import requests
from bs4 import BeautifulSoup

class SerperDevTool:
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.url = "https://google.serper.dev/search"
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not set in environment variables.")

    def search(self, query):
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query}
        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def extract_web_content(url, max_chars=2000):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Remove script/style
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            # Extract visible text
            text = ' '.join(soup.stripped_strings)
            return text[:max_chars]
        except Exception as e:
            return f"[ERROR extracting content: {e}]"