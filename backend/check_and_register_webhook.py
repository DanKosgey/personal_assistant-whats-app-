#!/usr/bin/env python3
import os, json, requests
import hashlib
import hmac
from dotenv import dotenv_values, load_dotenv

# Load .env files in a safe order so the script picks up values when run from
# different locations. Priority: repo root .env, agent/.env, backend/.env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
AGENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Load repo-level .env, then agent .env, then backend .env (do not override existing env vars)
try:
    load_dotenv(dotenv_path=os.path.join(REPO_ROOT, '.env'), override=False)
except Exception:
    pass
try:
    load_dotenv(dotenv_path=os.path.join(AGENT_DIR, '.env'), override=False)
except Exception:
    pass
try:
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'), override=False)
except Exception:
    pass

# Also provide a dict view of backend/.env for explicit fallbacks
cfg = dotenv_values(os.path.join(BASE_DIR, '.env'))

# Prefer environment variables, then values from backend/.env
WHATSAPP_ACCESS_TOKEN = os.getenv('WHATSAPP_ACCESS_TOKEN') or cfg.get('WHATSAPP_ACCESS_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID') or cfg.get('WHATSAPP_PHONE_NUMBER_ID')
WEBHOOK_VERIFY_TOKEN = os.getenv('WEBHOOK_VERIFY_TOKEN') or cfg.get('WEBHOOK_VERIFY_TOKEN')
APP_ID = os.getenv('APP_ID') or cfg.get('APP_ID')
APP_SECRET = os.getenv('APP_SECRET') or cfg.get('APP_SECRET')
APP_ACCESS_TOKEN = os.getenv('APP_ACCESS_TOKEN') or cfg.get('APP_ACCESS_TOKEN')

# If APP_ID and APP_SECRET are provided and APP_ACCESS_TOKEN isn't set or is empty, construct it
if APP_ID and APP_SECRET and (not APP_ACCESS_TOKEN or APP_ACCESS_TOKEN.strip() == ''):
    # app access token format: APP_ID|APP_SECRET
    APP_ACCESS_TOKEN = f"{APP_ID}|{APP_SECRET}"

# Compute appsecret_proof if we have both tokens and secret
APPSECRET_PROOF = None
if WHATSAPP_ACCESS_TOKEN and APP_SECRET:
    try:
        # For debug purposes
        print("\nGenerating appsecret_proof with:")
        print("WHATSAPP_ACCESS_TOKEN:", WHATSAPP_ACCESS_TOKEN)
        print("APP_SECRET:", APP_SECRET)
        
        # HMAC-SHA256 of the WhatsApp access token keyed by the app secret
        APPSECRET_PROOF = hmac.new(
            APP_SECRET.encode('utf-8'),
            WHATSAPP_ACCESS_TOKEN.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        print("Generated appsecret_proof:", APPSECRET_PROOF)
    except Exception as e:
        print("Error generating appsecret_proof:", str(e))
        APPSECRET_PROOF = None

print('--- Local env summary ---')
print('WHATSAPP_PHONE_NUMBER_ID:', WHATSAPP_PHONE_NUMBER_ID)
print('WEBHOOK_VERIFY_TOKEN:', WEBHOOK_VERIFY_TOKEN)

# 1) Query ngrok local API
ngrok_api = 'http://127.0.0.1:4040/api/tunnels'
ngrok_url = None
try:
    r = requests.get(ngrok_api, timeout=5)
    if r.status_code == 200:
        data = r.json()
        tunnels = data.get('tunnels', [])
        for t in tunnels:
            pu = t.get('public_url') or t.get('public_url')
            proto = t.get('proto') or t.get('protocol') or ''
            if pu and pu.startswith('https'):
                ngrok_url = pu
                break
        if not ngrok_url and tunnels:
            ngrok_url = tunnels[0].get('public_url')
    else:
        print('ngrok API returned', r.status_code)
except Exception as e:
    print('ngrok API not available or error:', e)

print('\nngrok public URL:', ngrok_url)

if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
    print('\nMissing WhatsApp credentials in .env; cannot query Graph API or register webhook')
    exit(2)

headers = {'Authorization': f'Bearer {WHATSAPP_ACCESS_TOKEN}'}

# 2) Query existing webhook configuration
graph_url_base = f'https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}'
print('\n--- Querying Graph API for current webhook config ---')
phone_webhooks_supported = True
try:
    get_params = {}
    if APPSECRET_PROOF:
        get_params['appsecret_proof'] = APPSECRET_PROOF
    r = requests.get(graph_url_base + '/webhooks', headers=headers, params=get_params, timeout=10)
    print('GET', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
except Exception as e:
    print('Graph API GET error:', e)

# If we get an unknown path/components error (code 2500), the phone-number /webhooks
# edge may not be the correct place to register. Try an app-level subscription if we
# have APP_ID and credentials; otherwise print manual instructions.
if r is not None and r.status_code == 400:
    try:
        resp = r.json()
        err = resp.get('error', {})
        if err.get('code') == 2500:
            print('\nDetected Graph API 2500: path not supported for phone-number webhooks.')
            # Mark that the phone-number /webhooks edge is not supported so we skip
            # attempting to POST to the phone-number level webhook endpoint later.
            phone_webhooks_supported = False
            if APP_ID and (APP_ACCESS_TOKEN or APP_SECRET):
                print('Attempting app-level webhook subscription using APP_ID...')
                # Build app access token
                if not APP_ACCESS_TOKEN and APP_SECRET:
                    # app access token format app_id|app_secret
                    APP_ACCESS_TOKEN = f"{APP_ID}|{APP_SECRET}"

                app_sub_url = f'https://graph.facebook.com/v18.0/{APP_ID}/subscriptions'
                callback_url = ngrok_url.rstrip('/') + '/api/webhook' if ngrok_url else None
                if not callback_url:
                    print('No ngrok public URL available; cannot register app-level subscription callback_url.')
                else:
                    params = {
                        'access_token': APP_ACCESS_TOKEN,
                        'object': 'whatsapp_business_account',
                        'callback_url': callback_url,
                        'fields': 'messages',
                        'verify_token': WEBHOOK_VERIFY_TOKEN
                    }
                    try:
                        ar = requests.post(app_sub_url, data=params, params={'appsecret_proof': APPSECRET_PROOF} if APPSECRET_PROOF else None, timeout=10)
                        print('APP SUBSCRIBE POST', ar.status_code)
                        try:
                            print(json.dumps(ar.json(), indent=2))
                        except Exception:
                            print(ar.text)
                    except Exception as e:
                        print('App-level subscription error:', e)
            else:
                print('\nNo APP_ID or app credentials found in .env or environment.')
                print('To subscribe your app to WhatsApp webhooks, use the Facebook App dashboard or run:')
                print('\n  POST https://graph.facebook.com/v18.0/{APP_ID}/subscriptions')
                print('    params: object=whatsapp_business_account, callback_url=<your-callback>,')
                print('            fields=messages, verify_token=<your-verify-token>, access_token=<APP_ACCESS_TOKEN>')
                print('\nIf you have your App Secret, you can use access_token="{APP_ID}|{APP_SECRET}"')
                print('\nExample curl:')
                print('  curl -X POST "https://graph.facebook.com/v18.0/<APP_ID>/subscriptions" -d "object=whatsapp_business_account" -d "callback_url=https://yourdomain.com/api/webhook" -d "fields=messages" -d "verify_token=whatsapp_webhook_2025" -d "access_token=APP_ID|APP_SECRET"')
    except Exception:
        pass

if r is not None and r.status_code == 401:
    print('\nERROR: Graph API returned 401 - token invalid or expired.')
    print('Steps to refresh your WhatsApp access token:')
    print(' 1) Visit your Meta/WhatsApp developer dashboard and generate a new permanent token or re-authorize your app.')
    print(' 2) Update WHATSAPP_ACCESS_TOKEN in backend/.env with the new token.')
    print(' 3) Re-run this script after updating the token.')
    exit(3)

# 3) If ngrok url present, attempt to register it as webhook
if ngrok_url:
    callback = ngrok_url.rstrip('/') + '/api/webhook'
    print('\nAttempting to register callback:', callback)
    payload = {
        'messaging_product': 'whatsapp',
        'webhooks': {
            'url': callback,
            'events': ['messages']
        }
    }
    # If the GET earlier indicated the phone-number webhooks edge is not supported
    # (Graph API 2500), skip trying to POST to the phone-number /webhooks endpoint
    # because it will return the same 2500 error. The app-level subscription above
    # is the supported mechanism for receiving WhatsApp messages in that case.
    if not phone_webhooks_supported:
        print('\nSkipping phone-number /webhooks registration because Graph API reports it is not supported (2500).')
        print('If you need to change the callback URL, update your App subscription at /{APP_ID}/subscriptions or use the Facebook App dashboard.')
    else:
        try:
            post_params = {}
            if APPSECRET_PROOF:
                post_params['appsecret_proof'] = APPSECRET_PROOF
            r = requests.post(graph_url_base + '/webhooks', headers={**headers, 'Content-Type':'application/json'}, params=post_params if post_params else None, json=payload, timeout=10)
            print('POST', r.status_code)
            try:
                print(json.dumps(r.json(), indent=2))
            except Exception:
                print(r.text)
        except Exception as e:
            print('Graph API POST error:', e)
else:
    print('\nNo ngrok URL found; skipping registration')

print('\nDone')
