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
    host = os.environ.get("HOST", "127.0.0.1")
    workers = os.environ.get("WORKERS")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.server:app",
        "--host",
        host,
        "--port",
        port,
    ]
    if workers:
        cmd += ["--workers", str(workers)]
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


def _configure_ngrok_auth(ngrok_path: str, authtoken: str):
    try:
        cmd = [ngrok_path, "config", "add-authtoken", authtoken]
        print("Configuring ngrok authtoken...")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("Warning: unable to configure ngrok authtoken:", e)


def start_ngrok(ngrok_path: str):
    # Launch ngrok to forward port from environment
    port = os.environ.get("PORT", "8001")
    region = os.environ.get("NGROK_REGION")
    domain = os.environ.get("NGROK_DOMAIN")  # reserved/custom domain (paid)
    authtoken = os.environ.get("NGROK_AUTHTOKEN")

    if authtoken:
        _configure_ngrok_auth(ngrok_path, authtoken)

    cmd = [ngrok_path, "http", port]
    if region:
        # ngrok v3 flag
        cmd += ["--region", region]
    if domain:
        # ngrok v3 reserved domain flag
        cmd += ["--domain", domain]
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
    # Prefer server-side registrar if present, else fallback to legacy backend script
    script = None
    candidate_server = ROOT / "server" / "check_and_register_webhook.py"
    candidate_backend = ROOT / "backend" / "check_and_register_webhook.py"
    if candidate_server.exists():
        script = candidate_server
    elif candidate_backend.exists():
        script = candidate_backend
    else:
        print("Webhook registration script not found:", candidate_server, candidate_backend)
        return 1

    env = os.environ.copy()
    env["NGROK_PUBLIC_URL"] = ngrok_url

    print("Registering webhook using", script)
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
        use_ngrok = os.environ.get("USE_NGROK", "1").lower() in ("1", "true", "yes")
        env_mode = os.environ.get("ENV", "development").lower()
        # Enforce ngrok in production
        if env_mode == "production" and not use_ngrok:
            print("FATAL: USE_NGROK must be enabled in production. Set USE_NGROK=1 and configure NGROK_AUTHTOKEN and NGROK_DOMAIN.")
            sys.exit(2)

        # Validate ngrok envs
        port = os.environ.get("NGROK_PORT") or os.environ.get("PORT", "8001")
        os.environ.setdefault("NGROK_PORT", str(port))
        proto = os.environ.get("NGROK_PROTOCOL", "https").lower()
        region = os.environ.get("NGROK_REGION", "us")
        domain = os.environ.get("NGROK_DOMAIN")
        authtoken = os.environ.get("NGROK_AUTHTOKEN")

        if use_ngrok:
            if env_mode == "production" and not authtoken:
                print("FATAL: NGROK_AUTHTOKEN is required in production")
                sys.exit(2)
            if env_mode == "production" and not domain and os.environ.get("ALLOW_EPHEMERAL_NGROK", "0").lower() not in ("1","true","yes"):
                print("FATAL: NGROK_DOMAIN is required in production for a stable reserved hostname. Set ALLOW_EPHEMERAL_NGROK=1 only for development.")
                sys.exit(2)
        if use_ngrok:
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
                    port = os.environ.get("NGROK_PORT") or os.environ.get("PORT", "8001")
                    authtoken = os.environ.get("NGROK_AUTHTOKEN")
                    if authtoken:
                        _ngrok.set_auth_token(authtoken)
                    region = os.environ.get("NGROK_REGION")
                    protocol = os.environ.get("NGROK_PROTOCOL", "https").lower()
                    opts = {"addr": str(port), "proto": protocol}
                    if region:
                        opts["region"] = region
                    domain = os.environ.get("NGROK_DOMAIN")
                    if domain:
                        opts["domain"] = domain
                    print(f"Starting pyngrok tunnel on port {port}...")
                    public_url = str(_ngrok.connect(**opts))
                    used_pyngrok = True
                    print("ngrok public URL:", public_url)
                except Exception as e:
                    print("pyngrok fallback failed:", e)
        else:
            print("USE_NGROK is disabled; skipping tunnel startup")

        # If we have a public URL, run webhook registration
        if public_url:
            os.environ["PUBLIC_WEBHOOK_URL"] = public_url.rstrip('/')
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
