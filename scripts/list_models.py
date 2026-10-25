import os

import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("LLM_API_KEY", "")
if not key:
    raise SystemExit("LLM_API_KEY is empty in .env")

r = httpx.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    headers={"x-goog-api-key": key},
    params={"pageSize": 100},
    timeout=30,
)
r.raise_for_status()
for m in r.json().get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"].replace("models/", ""))