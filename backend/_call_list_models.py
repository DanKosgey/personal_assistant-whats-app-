import os
import json
import traceback
from dotenv import load_dotenv
import google.generativeai as genai

# Load backend/.env so GEMINI_API_KEY is available when running from the backend folder
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
    print('GEMINI_API_KEY not set in environment or .env')
    raise SystemExit(2)

print('Using GEMINI primary key prefix:', API_KEY[:8] + '...')

try:
    genai.configure(api_key=API_KEY)
    print('Configured genai')
    resp = genai.list_models()
    print('list_models() returned type:', type(resp))
    try:
        # try to convert to json if possible
        print('repr(resp):', repr(resp)[:1000])
    except Exception:
        pass
    # inspect common attributes
    if hasattr(resp, 'models'):
        print('resp.models type:', type(resp.models))
        try:
            print('first 10 models:')
            for m in resp.models[:10]:
                print(' -', getattr(m, 'name', m))
        except Exception as e:
            print('Could not iterate resp.models:', e)
    elif isinstance(resp, (list, tuple)):
        print('resp is list/tuple, len:', len(resp))
        for m in resp[:20]:
            print(' -', getattr(m, 'name', m))
    else:
        # print keys if dict-like
        try:
            items = dict(resp)
            print('dict(resp) keys:', list(items.keys())[:50])
        except Exception:
            pass

except Exception as e:
    print('Exception when calling list_models():')
    traceback.print_exc()
    raise
