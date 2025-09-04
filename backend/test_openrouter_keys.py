import os
import asyncio
import logging
import argparse
from openrouter_client import OpenRouterClient, extract_text_from_openrouter_response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_openrouter_keys")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run in offline simulation mode (no network calls).")
    args, _ = parser.parse_known_args()
    # Load keys from env; if none found, try to load backend/.env automatically
    def try_load_dotenv(path):
        if not os.path.exists(path):
            return 0
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
                    loaded += 1
        return loaded

    keys = []
    for k, v in sorted(os.environ.items()):
        if k.upper().startswith("OPENROUTER_API_KEY") and v:
            keys.append((k, v.strip()))

    if not keys:
        # try to load .env from this directory
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        loaded = try_load_dotenv(env_path)
        if loaded:
            print(f"Loaded {loaded} entries from {env_path}")
        # re-scan
        keys = []
        for k, v in sorted(os.environ.items()):
            if k.upper().startswith("OPENROUTER_API_KEY") and v:
                keys.append((k, v.strip()))

    if not keys:
        print("No OPENROUTER_API_KEY_* variables found in environment.")
        return

    model = os.getenv("OPENROUTER_MODEL_ID")
    print("Using model:", model)
    client = OpenRouterClient(model=model, keys=[v for _, v in keys])

    # test each key individually
    results = []
    for name, key in keys:
        print(f"Testing {name} ...", end=" ", flush=True)
        if args.offline:
            # offline simulation: mark as simulated OK
            print("SIMULATED-OK: (offline mode)")
            results.append((name, "simulated-ok", "offline"))
            continue
        try:
            data = await client.test_key(key, model=model)
            text = extract_text_from_openrouter_response(data)
            print("OK:", (text or "").strip()[:200])
            results.append((name, "ok", text))
        except Exception as e:
            print("ERROR:", str(e))
            results.append((name, "error", str(e)))

    # summary
    print("\nSummary:")
    for name, status, info in results:
        print(f" - {name}: {status} -> {info[:200] if isinstance(info, str) else 'response object'}")


if __name__ == "__main__":
    asyncio.run(main())
