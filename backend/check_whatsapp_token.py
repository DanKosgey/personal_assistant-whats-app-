#!/usr/bin/env python3
"""Simple helper to validate the WHATSAPP_ACCESS_TOKEN and report Graph API errors.
Run from the backend folder where `.env` lives.
"""
import os, sys, json
from dotenv import dotenv_values
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cfg = dotenv_values(os.path.join(BASE_DIR, '.env'))

WHATSAPP_ACCESS_TOKEN = os.getenv('WHATSAPP_ACCESS_TOKEN') or cfg.get('WHATSAPP_ACCESS_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID') or cfg.get('WHATSAPP_PHONE_NUMBER_ID')

if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
    print('Missing WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID in environment/.env')
    sys.exit(2)

url = f'https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}?fields=id'
headers = {'Authorization': f'Bearer {WHATSAPP_ACCESS_TOKEN}'}
print('Checking token against:', url)
try:
    r = requests.get(url, headers=headers, timeout=10)
    print('HTTP', r.status_code)
    try:
        data = r.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print('Non-JSON response:')
        print(r.text)

    if r.status_code == 200:
        print('\nToken is valid for phone number id. Webhook registration and message sending should work.')
        sys.exit(0)
    else:
        err = None
        try:
            err = r.json().get('error', {})
        except Exception:
            pass
        if err:
            code = err.get('code')
            subcode = err.get('error_subcode')
            msg = err.get('message')
            print('\nGraph API error details:')
            print(' code:', code)
            print(' subcode:', subcode)
            print(' message:', msg)
            if code == 190:
                print('\nThis indicates an invalid or expired access token. To fix:')
                print(' 1) Generate a new access token in the Meta developer dashboard or the WhatsApp Business Cloud console.')
                print(' 2) Replace WHATSAPP_ACCESS_TOKEN in backend/.env with the new token.')
                print(' 3) Re-run this script and then re-run the webhook registration helper.')
        else:
            print('\nUnknown non-200 response. Inspect the HTTP body above for details.')
except Exception as e:
    print('Request error:', e)
    sys.exit(3)
