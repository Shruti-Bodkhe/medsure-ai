import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "")

TIMEOUT = httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=20.0)
RETRIES = 3


def _post(url: str, body: dict, headers: dict) -> httpx.Response:
    """POST with retries on timeouts, rate limits (429) and server errors (5xx)."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                return r
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < RETRIES:
            wait = 5 * attempt
            print(f"[llm] attempt {attempt} failed ({last}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM request failed after {RETRIES} attempts. Last error: {last}")


def generate(prompt: str, system: str = "", temperature: float = 0.0,
             max_tokens: int = 800) -> str:
    """Send one prompt to the configured LLM and return the text reply."""
    if not API_KEY or not MODEL:
        raise RuntimeError("Set LLM_API_KEY and LLM_MODEL in .env")

    if PROVIDER == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = _post(url, body, {"x-goog-api-key": API_KEY})
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:400]}")
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Gemini returned no text (maybe blocked): {str(data)[:400]}")
        return "".join(p.get("text", "") for p in parts).strip()

    if PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = {"model": MODEL, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        r = _post(url, body, {"Authorization": f"Bearer {API_KEY}"})
        if r.status_code >= 400:
            raise RuntimeError(f"Groq error {r.status_code}: {r.text[:400]}")
        return r.json()["choices"][0]["message"]["content"].strip()

    raise RuntimeError(f"Unknown LLM_PROVIDER: {PROVIDER}")