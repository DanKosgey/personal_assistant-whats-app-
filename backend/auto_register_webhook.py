"""Auto-register webhook helper
Polls local ngrok API until a public https URL is available, then calls the
existing check_and_register_webhook logic to register the callback URL.

Usage: python auto_register_webhook.py
"""
import time
import requests
import subprocess
import os
from dotenv import dotenv_values

cfg = dotenv_values('.env')

NGROK_API = 'http://127.0.0.1:4040/api/tunnels'

print('Waiting for ngrok public URL...')
for i in range(60):
    try:
        r = requests.get(NGROK_API, timeout=5)
        if r.status_code == 200:
            data = r.json()
            tunnels = data.get('tunnels', [])
            for t in tunnels:
                pu = t.get('public_url')
                if pu and pu.startswith('https'):
                    print('Found public URL:', pu)
                    # Call the existing registration script
                    subprocess.run(['python', 'check_and_register_webhook.py'], check=False)
                    raise SystemExit(0)
    except Exception as e:
        pass
    time.sleep(2)

print('No public ngrok URL found after waiting; please start ngrok and try again.')
