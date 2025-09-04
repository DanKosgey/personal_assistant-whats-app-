"""List available Generative models using Google Generative Language REST API and try a short generation.

Requires: GEMINI_API_KEY in environment or backend/.env
This script uses direct HTTPS calls with the API key to avoid Application Default Credentials issues.
"""
import os
import sys
import json
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
    print('GEMINI_API_KEY not found in environment or .env; please set GEMINI_API_KEY or configure gemini_api_keys in server.AgentConfig')
    if not dev:
        sys.exit(2)
    else:
        print('DEV_SMOKE enabled; continuing in dev mode (API calls will be skipped)')

BASE = 'https://generativelanguage.googleapis.com/v1'

def list_models():
    url = f"{BASE}/models?key={API_KEY}"
    print('GET', url)
    r = requests.get(url, timeout=30)
    try:
        r.raise_for_status()
    except Exception as e:
        print('List models request failed:', r.status_code, r.text)
        raise
    data = r.json()
    models = data.get('models') or []
    print(f"Found {len(models)} models")
    for m in models:
        # model may have 'name' and 'displayName'
        print('-', m.get('name') or m.get('displayName') or m)
    return models


def generate_with_model(model_name, prompt='Say "Test successful" if you can respond.', max_tokens=200):
    # model_name may be full path like 'models/text-bison-001' or short 'text-bison-001'
    if not model_name.startswith('models/'):
        model_path = f"models/{model_name}"
    else:
        model_path = model_name
    url = f"{BASE}/{model_path}:generateText?key={API_KEY}"
    payload = {
        'prompt': {'text': prompt},
        'temperature': 0.2,
        'maxOutputTokens': max_tokens,
    }
    print('POST', url)
    r = requests.post(url, json=payload, timeout=30)
    try:
        r.raise_for_status()
    except Exception:
        print('Generation request failed:', r.status_code, r.text)
        raise
    return r.json()


def main():
    models = list_models()
    if not models:
        print('No models returned by REST API.')
        return 1
    # pick candidate model heuristically
    chosen = None
    for m in models:
        name = m.get('name','')
        if any(x in name.lower() for x in ('bison','gemini','chat','text')):
            chosen = name
            break
    if not chosen:
        chosen = models[0].get('name')
    print('\nChosen model:', chosen)
    try:
        res = generate_with_model(chosen)
        print('\nGeneration response (truncated):')
        print(json.dumps(res, indent=2)[:4000])
    except Exception as e:
        print('Generation failed:', e)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
