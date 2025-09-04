import json
from pathlib import Path
from typing import Dict

CTX_PATH = Path(__file__).parent / 'context.json'


def load_context() -> Dict[str, str]:
    try:
        with CTX_PATH.open('r', encoding='utf-8') as f:
            data = json.load(f)
            return {k: str(v) for k, v in data.items()}
    except Exception:
        return {}
