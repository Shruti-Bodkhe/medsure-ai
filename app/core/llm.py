import base64
import hashlib
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "")
FALLBACKS = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]

TIMEOUT = httpx.Timeout(connect=20.0, read=25.0, write=20.0, pool=20.0)
RETRIES = 2

LAST_MODEL_USED = ""
_CACHE = {}
_DISCOVERED_MODELS = None


def _get_available_gemini_models() -> list:
    """Dynamically fetch valid generateContent models for this API key so 404s never happen."""
    global _DISCOVERED_MODELS
    if _DISCOVERED_MODELS is not None:
        return _DISCOVERED_MODELS
    try:
        r = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": API_KEY},
            params={"pageSize": 100},
            timeout=15.0,
        )
        if r.status_code == 200:
            all_models = []
            for m in r.json().get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    name = m["name"].replace("models/", "")
                    # Skip vision-only, embedding, or tts-only specialized endpoints
                    if any(skip in name for skip in ("tts", "image", "embedding", "aqa", "bisheng", "thinking")):
                        continue
                    all_models.append(name)
            # Prioritize flash models first for speed & quota
            flash_models = [m for m in all_models if "flash" in m]
            other_models = [m for m in all_models if "flash" not in m]
            _DISCOVERED_MODELS = flash_models + other_models
            return _DISCOVERED_MODELS
    except Exception:
        pass
    _DISCOVERED_MODELS = []
    return _DISCOVERED_MODELS


def _candidate_models() -> list:
    configured = [m for m in [MODEL] + FALLBACKS if m]
    if PROVIDER == "gemini":
        discovered = _get_available_gemini_models()
        # Keep configured models that actually exist on the account, then append other discovered models
        valid_configured = [m for m in configured if not discovered or m in discovered]
        extras = [m for m in discovered if m not in valid_configured]
        return (valid_configured + extras)[:5] or configured
    return configured or ["llama-3.3-70b-versatile"]


def _post(url: str, body: dict, headers: dict) -> httpx.Response:
    """POST with fast retries on timeouts, rate limits (429) and server errors (5xx)."""
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
            wait = 2 * attempt
            print(f"[llm] attempt {attempt} failed ({last[:80]}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM request failed after {RETRIES} attempts. Last error: {last}")


def _gemini(model: str, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
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
        raise RuntimeError(f"Gemini returned no text: {str(data)[:400]}")
    return "".join(p.get("text", "") for p in parts).strip()


def _groq(model: str, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    r = _post(url, body, {"Authorization": f"Bearer {API_KEY}"})
    if r.status_code >= 400:
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:400]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def generate(prompt: str, system: str = "", temperature: float = 0.0,
             max_tokens: int = 800) -> str:
    """Send one prompt to the configured LLM; fall back to backup models if busy."""
    global LAST_MODEL_USED
    if not API_KEY:
        raise RuntimeError("Set LLM_API_KEY in .env")
    if PROVIDER not in {"gemini", "groq"}:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {PROVIDER}")

    cache_key = hashlib.md5(f"{PROVIDER}:{system}:{prompt}:{temperature}".encode("utf-8")).hexdigest()
    if temperature == 0.0 and cache_key in _CACHE:
        return _CACHE[cache_key]

    call = _gemini if PROVIDER == "gemini" else _groq
    models_to_try = _candidate_models()

    errors = []
    for model in models_to_try:
        try:
            text = call(model, prompt, system, temperature, max_tokens)
            LAST_MODEL_USED = model
            if temperature == 0.0:
                _CACHE[cache_key] = text
            return text
        except RuntimeError as e:
            errors.append(f"{model}: {str(e)[:150]}")
    raise RuntimeError("All models failed:\n  " + "\n  ".join(errors))


def generate_multimodal(prompt: str, image_data: bytes, mime_type: str = "image/jpeg",
                        max_tokens: int = 1500) -> str:
    """Analyze medical images, prescriptions, and lab reports using multimodal vision."""
    global LAST_MODEL_USED
    b64_image = base64.b64encode(image_data).decode("utf-8")

    if PROVIDER == "gemini":
        for model in _candidate_models():
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                body = {
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime_type, "data": b64_image}}
                        ]
                    }],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens}
                }
                r = _post(url, body, {"x-goog-api-key": API_KEY})
                if r.status_code < 400:
                    data = r.json()
                    parts = data["candidates"][0]["content"]["parts"]
                    LAST_MODEL_USED = model
                    return "".join(p.get("text", "") for p in parts).strip()
            except Exception:
                continue

    elif PROVIDER == "groq":
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            body = {
                "model": "llama-3.2-11b-vision-preview",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
                    ]
                }],
                "temperature": 0.1,
                "max_tokens": max_tokens
            }
            r = _post(url, body, {"Authorization": f"Bearer {API_KEY}"})
            if r.status_code < 400:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    raise RuntimeError("Multimodal vision analysis failed across configured models.")