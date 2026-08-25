"""Minimal chat-completions client: key handling, retries, on-disk cache.

Provider-agnostic, because any OpenAI-compatible endpoint will do and the
project has already outrun one monthly quota. Select with PROVIDER or the
--provider flag of the calling script; keys live in phdq_ext/.env, which is
gitignored, and are never logged or written into results.
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
USAGE_LOG = os.path.join(BASE, "results", "api_usage.jsonl")

PROVIDERS = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "env": "OPENROUTER_API_KEY"},
    "apiyi": {"url": "https://api.apiyi.com/v1/chat/completions",
              "env": "APIYI_API_KEY"},
    "openai": {"url": "https://api.openai.com/v1/chat/completions",
               "env": "OPENAI_API_KEY"},
}
PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")
API_URL = PROVIDERS[PROVIDER]["url"]


def get_key(provider=None):
    name = PROVIDERS[provider or PROVIDER]["env"]
    key = os.environ.get(name)
    if key:
        return key.strip()
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith(name):
                    return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f"no {name}: set it in the environment or in {ENV_PATH}")


def probe_cost(prompt, model, provider=None, **kw):
    """One call, reporting the tokens it actually consumed.

    Run this before any batch. A reasoning model can answer a 600-token request
    with tens of thousands of tokens of hidden thinking, which does not appear
    in the reply and does appear on the bill: a 442-call batch on a flash-tier
    model cost several times what 1200 calls on a frontier model did, purely
    from that. The reply length tells you nothing; the usage block does.
    """
    out, usage = complete(prompt, model, provider=provider, use_cache=False,
                          return_usage=True, **kw)
    print(f"{model}: {usage}; ответ {len(out.split())} слов")
    return usage


def complete(prompt, model, temperature=0.7, max_tokens=2048, system=None,
             retries=4, timeout=180, use_cache=True, provider=None,
             thinking=False, return_usage=False):
    """One chat completion. Cached on disk by (model, system, prompt, temp).

    thinking=False asks the provider to switch reasoning off in each of the
    spellings the OpenAI-compatible gateways accept. A gateway that ignores
    all of them will still bill for the thinking, so check with probe_cost
    before launching a batch.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    # the provider is not part of the cache key: the same model through a
    # different gateway is the same request
    key_material = json.dumps(
        [model, system, prompt, temperature, max_tokens], sort_keys=True
    )
    cache_path = os.path.join(
        CACHE_DIR, hashlib.md5(key_material.encode()).hexdigest() + ".json"
    )
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            hit = json.load(f)
        return ((hit["text"], hit.get("usage", {})) if return_usage
                else hit["text"])

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if not thinking:
        body["reasoning_effort"] = "none"
        body["thinking"] = {"type": "disabled"}
        body["extra_body"] = {"google": {"thinking_config":
                                         {"thinking_budget": 0}}}
    payload = json.dumps(body).encode()

    last = None
    for attempt in range(retries):
        prov = provider or PROVIDER
        req = urllib.request.Request(
            PROVIDERS[prov]["url"],
            data=payload,
            headers={
                "Authorization": f"Bearer {get_key(prov)}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                rb = json.loads(resp.read().decode())
            text = rb["choices"][0]["message"]["content"]
            usage = rb.get("usage", {})
            try:
                os.makedirs(os.path.dirname(USAGE_LOG), exist_ok=True)
                with open(USAGE_LOG, "a") as f:
                    f.write(json.dumps({"model": model, "provider": prov,
                                        **usage}) + "\n")
            except OSError:
                pass
            if use_cache:
                with open(cache_path, "w") as f:
                    json.dump({"model": model, "text": text, "usage": usage},
                              f)
            return (text, usage) if return_usage else text
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
                    raise RuntimeError(f"{prov} {exc.code}: {detail}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{prov} failed after {retries} attempts: {last}")


def check():
    """Cheap end-to-end check that the key works."""
    return complete("Reply with the single word: ok",
                    model="openai/gpt-4o-mini", max_tokens=10,
                    temperature=0, use_cache=False)
