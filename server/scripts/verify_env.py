import os

def check_env(k):
    v = os.getenv(k)
    return 'SET' if v else 'NOT SET'

keys = [
    'MONGO_URL',
    'REDIS_URL',
    'GEMINI_API_KEYS',
    'GEMINI_API_KEY',
    'OPENROUTER_API_KEYS',
    'OPENROUTER_API_KEY',
    'WHATSAPP_ACCESS_TOKEN',
    'WHATSAPP_PHONE_NUMBER_ID',
    'DEV_SMOKE',
    'DISABLE_WHATSAPP_SENDS',
    'USE_INMEMORY_CACHE'
]

for k in keys:
    print(f"{k}: {check_env(k)}")
