from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

print('Configured genai? attempting to list')
# assume genai is already configured via GEMINI_API_KEY in env
try:
    gen = genai.list_models()
    n = 0
    for item in gen:
        name = getattr(item, 'name', None)
        if not name:
            try:
                # try dict-like
                name = item.get('name')
            except Exception:
                name = repr(item)
        print('MODEL:', name)
        n += 1
        if n >= 50:
            break
    if n == 0:
        print('No models returned')
except Exception as e:
    import traceback
    print('Exception while iterating list_models:')
    traceback.print_exc()
