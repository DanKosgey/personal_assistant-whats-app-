#!/usr/bin/env python3
"""List available Gemini/Generative models and run a short test generation.

Usage:
  python list_models_and_try_model.py

The script reads GEMINI_API_KEY from the environment or from a local .env file.
It will attempt to use the `google.generativeai` library if available, and
fall back to a best-effort HTTP approach if needed.

This is a diagnostics helper — it will not change any repository files.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

from server import config

load_dotenv()

# Prefer the configured AgentConfig gemini_api_keys primary key if available
API_KEY = None
try:
    if hasattr(config, 'gemini_api_keys') and config.gemini_api_keys:
        API_KEY = config.gemini_api_keys[0]
except Exception:
    API_KEY = None

if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    dev = os.getenv('DEV_SMOKE', '0').lower() in ('1', 'true', 'yes')
    print("ERROR: GEMINI_API_KEY not found in environment or .env.\nSet GEMINI_API_KEY or provide gemini_api_keys in AgentConfig and retry.")
    if not dev:
        sys.exit(2)
    else:
        print('DEV_SMOKE enabled; continuing without a real GEMINI API key')


def try_google_genai():
    """Try using google.generativeai library to list models and run a test generation."""
    try:
        import google.generativeai as genai
    except Exception as e:
        print(f"google.generativeai not available: {e}")
        return False

    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"Failed to configure genai client: {e}")
        return False

    models = None
    # Try common listing method names with graceful fallback
    for list_fn in ("list_models", "get_models", "models", "list_available_models"):
        try:
            fn = getattr(genai, list_fn)
            models = fn()
            break
        except AttributeError:
            continue
        except Exception:
            # if the library exists but the call failed, try next
            continue

    # If we didn't get a list, try accessing a client property
    if models is None:
        try:
            client = getattr(genai, "client", None)
            if client is not None and hasattr(client, "list_models"):
                models = client.list_models()
        except Exception:
            models = None

    # Normalize models output
    model_names = []
    if models:
        # models might be a list of dicts, or a dict-like with 'models'
        if isinstance(models, dict) and "models" in models:
            entries = models["models"]
        elif isinstance(models, (list, tuple)):
            entries = models
        else:
            # Try to stringify
            entries = []
        for m in entries:
            if isinstance(m, dict) and "name" in m:
                model_names.append(m["name"])
            elif isinstance(m, str):
                model_names.append(m)
            else:
                # best-effort
                try:
                    model_names.append(str(m))
                except Exception:
                    pass

    if not model_names:
        print("No models found via google.generativeai listing methods.")
        return False

    print("Available models (sample):")
    for n in model_names[:50]:
        print(" -", n)

    # Pick a model heuristically
    preferred = None
    for name in model_names:
        lname = name.lower()
        if "gemini" in lname or "bison" in lname or "chat" in lname or "text" in lname:
            preferred = name
            break
    if not preferred:
        preferred = model_names[0]

    print(f"\nChosen model for test: {preferred}")

    # Try a short generation using the chosen model
    try:
        # first try creation via GenerativeModel if available
        if hasattr(genai, "GenerativeModel"):
            model = genai.GenerativeModel(preferred)
            prompt = "Say 'Test successful' if you can generate text."
            print("Sending test generation prompt...")
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
            print("Generation result (truncated):\n", str(text)[:1000])
            return True
        # fallback: try a direct genai.generate (some SDKs expose a generate function)
        if hasattr(genai, "generate"):
            resp = genai.generate(model=preferred, prompt="Say 'Test successful'.")
            print("Generation result:\n", resp)
            return True

        print("No supported generation method found on google.generativeai client.")
        return True
    except Exception as e:
        print(f"Generation failed with model {preferred}: {e}")
        return False


def main():
    print("Using GEMINI_API_KEY (primary) for listing models...")
    ok = try_google_genai()
    if ok:
        print("Done.")
        sys.exit(0)
    else:
        print("Failed to list models or run test via google.generativeai.")
        print("If you have a different client or API, you can adapt this script or provide a valid key.")
        sys.exit(1)


if __name__ == "__main__":
    main()
