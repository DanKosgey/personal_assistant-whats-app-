"""Launcher: start the server, start ngrok, and register webhooks automatically.

Usage: python run_with_ngrok.py

Behavior:
- Starts the FastAPI app using uvicorn in a subprocess
- Starts local ngrok (looks for ngrok.exe under the workspace root)
- Polls the ngrok local API until a public https URL is available
- Calls the existing backend/check_and_register_webhook.py to register the callback
- Cleans up child processes on exit
"""
import os
import sys
import time
import signal
import subprocess
import requests
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Use repository root (directory where this script lives)
WORKSPACE_ROOT = ROOT
from dotenv import load_dotenv

# Load repo .env into this process so subprocesses inherit credentials when run_webhook_registration
try:
    load_dotenv(dotenv_path=WORKSPACE_ROOT / '.env', override=False)
except Exception:
    pass


def find_ngrok():
    candidates = [
        WORKSPACE_ROOT / "ngrok",
        WORKSPACE_ROOT / "ngrok.exe",
        WORKSPACE_ROOT / "ngrok" / "ngrok",
        WORKSPACE_ROOT / "ngrok" / "ngrok.exe",
        WORKSPACE_ROOT / "tools" / "ngrok",
        WORKSPACE_ROOT / "tools" / "ngrok.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def start_uvicorn():
    # Start uvicorn as a subprocess using the local python interpreter
    port = os.environ.get("PORT", "8001")
    cmd = [sys.executable, "-m", "uvicorn", "server.server:app", "--host", "127.0.0.1", "--port", port]
    print("Starting uvicorn:", " ".join(cmd))
    env = os.environ.copy()
    # Ensure imports resolve from repo root
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = str(ROOT)
    # Safe dev defaults unless explicitly overridden
    env.setdefault("DISABLE_DB", "1")
    env.setdefault("DEV_SMOKE", "1")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=None, stderr=None, env=env)


def start_ngrok(ngrok_path: str):
    # Launch ngrok to forward port from environment
    port = os.environ.get("PORT", "8001")
    cmd = [ngrok_path, "http", port]
    print("Starting ngrok:", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(WORKSPACE_ROOT), stdout=None, stderr=None)


def wait_for_ngrok_public_url(timeout: int = 60):
    api = "http://127.0.0.1:4040/api/tunnels"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(api, timeout=3)
            if r.status_code == 200:
                data = r.json()
                tunnels = data.get("tunnels", [])
                for t in tunnels:
                    pu = t.get("public_url")
                    if pu and pu.startswith("https"):
                        return pu
        except Exception:
            pass
        time.sleep(1)
    return None


def run_webhook_registration(ngrok_url: str):
    """Run the existing registration script from backend/."""
    script = ROOT / "backend" / "check_and_register_webhook.py"
    if not script.exists():
        print("Webhook registration script not found:", script)
        return 1

    env = os.environ.copy()
    env["NGROK_PUBLIC_URL"] = ngrok_url

    print("Registering webhook using backend/check_and_register_webhook.py")
    res = subprocess.run([sys.executable, str(script)], cwd=str(script.parent), env=env)
    return res.returncode


def main():
    ngrok_path = find_ngrok()
    if not ngrok_path:
        print("ngrok executable not found in workspace. Looking for pyngrok fallback...")

    # Start uvicorn
    uvicorn_proc = start_uvicorn()

    ngrok_proc = None
    public_url = None
    used_pyngrok = False

    try:
        if ngrok_path:
            ngrok_proc = start_ngrok(ngrok_path)
            print("Waiting for ngrok to publish a public URL...")
            public_url = wait_for_ngrok_public_url(timeout=60)
            if public_url:
                print("ngrok public URL:", public_url)
            else:
                print("ngrok did not publish a public HTTPS URL within timeout")
        else:
            try:
                from pyngrok import ngrok as _ngrok
                port = os.environ.get("PORT", "8001")
                print(f"Starting pyngrok tunnel on port {port}...")
                public_url = str(_ngrok.connect(str(port)))
                used_pyngrok = True
                print("ngrok public URL:", public_url)
            except Exception as e:
                print("pyngrok fallback failed:", e)

        # If we have a public URL, run webhook registration
        if public_url:
            code = run_webhook_registration(public_url)
            if code != 0:
                print("Webhook registration script exited with code:", code)
        else:
            print("Skipping webhook registration (no public URL)")

        print("Server is running. Press Ctrl-C to stop.")

        # Wait for uvicorn to exit (block)
        uvicorn_proc.wait()

    except KeyboardInterrupt:
        print("Interrupted, shutting down...")
    finally:
        # Cleanup
        # Cleanup ngrok/pyngrok
        if used_pyngrok:
            try:
                from pyngrok import ngrok as _ngrok
                _ngrok.kill()
            except Exception:
                pass
        for p in (ngrok_proc, uvicorn_proc):
            try:
                if p and p.poll() is None:
                    p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
