"""Try generateText on available models (REST) until one succeeds.

Reads GEMINI_API_KEY from .env or environment.
Prints per-model result and exits when a generation succeeds.
"""
import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

from server import config

load_dotenv()
API_KEY = None
try:
    if hasattr(config, 'gemini_api_keys') and config.gemini_api_keys:
        API_KEY = config.gemini_api_keys[0]
except Exception:
    API_KEY = None

if not API_KEY:
    API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    dev = os.getenv('DEV_SMOKE', '0').lower() in ('1', 'true', 'yes')
    print('GEMINI_API_KEY not found in environment or .env; set GEMINI_API_KEY or configure gemini_api_keys in server.AgentConfig')
    if not dev:
        sys.exit(2)
    else:
        print('DEV_SMOKE enabled; continuing in dev mode (no real API calls will be made)')

BASE = 'https://generativelanguage.googleapis.com/v1'

def list_models():
    r = requests.get(f"{BASE}/models?key={API_KEY}", timeout=30)
    r.raise_for_status()
    return r.json().get('models', [])

def try_generate(model_name):
    if model_name.startswith('models/'):
        model_path = model_name
    else:
        model_path = f"models/{model_name}"
    url = f"{BASE}/{model_path}:generateText?key={API_KEY}"
    payload = {
        'prompt': {'text': 'Test generation: say Test successful.'},
        'temperature': 0.2,
        'maxOutputTokens': 200,
    }
    r = requests.post(url, json=payload, timeout=30)
    return r


def main():
    models = list_models()
    if not models:
        print('No models found')
        return 1
    tried = 0
    for m in models:
        name = m.get('name') or m.get('displayName') or str(m)
        print('\nTrying model:', name)
        # skip obvious embedding-only models
        if 'embedding' in name.lower():
            print(' - skipping embedding model')
            continue
        try:
            r = try_generate(name)
            print(' - status:', r.status_code)
            if r.status_code == 200:
                j = r.json()
                print(' - Success. Response snippet:')
                print(json.dumps(j, indent=2)[:1500])
                return 0
            else:
                print(' - Failed. body:', r.text[:1000])
        except Exception as e:
            print(' - Exception:', e)
        tried += 1
        if tried >= 8:
            print('Tried 8 models, stopping to avoid quota/slowdowns')
            break
        time.sleep(0.5)
    print('No model succeeded for generateText with this API key')
    return 2

if __name__ == '__main__':
    sys.exit(main())
