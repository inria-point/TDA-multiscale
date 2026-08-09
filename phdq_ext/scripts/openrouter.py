"""Minimal OpenRouter client: key handling, retries, on-disk response cache.

The key is read from the environment or from phdq_ext/.env, which is
gitignored. It is never logged or written into results.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

BASE = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(BASE, "cache", "openrouter")
ENV_PATH = os.path.join(BASE, ".env")
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY"):
                    return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(
        f"no OPENROUTER_API_KEY: set it in the environment or in {ENV_PATH}"
    )


def complete(prompt, model, temperature=0.7, max_tokens=2048, system=None,
             retries=4, timeout=180, use_cache=True):
    """One chat completion. Cached on disk by (model, system, prompt, temp)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key_material = json.dumps(
        [model, system, prompt, temperature, max_tokens], sort_keys=True
    )
    cache_path = os.path.join(
        CACHE_DIR, hashlib.md5(key_material.encode()).hexdigest() + ".json"
    )
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)["text"]

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens}
    ).encode()

    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {get_key()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
            text = body["choices"][0]["message"]["content"]
            if use_cache:
                with open(cache_path, "w") as f:
                    json.dump({"model": model, "text": text}, f)
            return text
        except (urllib.error.HTTPError, urllib.error.URLError,
                KeyError, TimeoutError) as exc:
            last = exc
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode()[:200]
                except Exception:
                    pass
                if exc.code in (400, 401, 403):  # not worth retrying
                    raise RuntimeError(f"OpenRouter {exc.code}: {detail}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter failed after {retries} attempts: {last}")


def check():
    """Cheap end-to-end check that the key works."""
    return complete("Reply with the single word: ok",
                    model="openai/gpt-4o-mini", max_tokens=10,
                    temperature=0, use_cache=False)
