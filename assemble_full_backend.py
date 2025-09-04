"""
Assemble full backend into whats_app_agent/backend by copying files from agent/whats-app-agent/backend.
Excludes: .env, .venv, __pycache__, *.pyc, large files >5MB, .git
Produces: whats_app_agent/backend/.env.example and whats_app_agent/report.json summary.
Run from repository root: python whats_app_agent/assemble_full_backend.py
"""
import os, shutil, json, stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'agent' / 'whats-app-agent' / 'backend'
DST = Path(__file__).resolve().parent / 'backend'

EXCLUDE_NAMES = {'.env', '.venv', '__pycache__', '.git'}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo', '.dist-info', '.egg-info'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

copied = []
excluded = []
errors = []

def should_exclude(path: Path):
    name = path.name
    if name in EXCLUDE_NAMES:
        return True
    if any(part in EXCLUDE_NAMES for part in path.parts):
        return True
    if any(name.endswith(s) for s in EXCLUDE_SUFFIXES):
        return True
    try:
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE:
            return True
    except Exception:
        return False
    return False

if not SRC.exists():
    print('Source backend not found at', SRC)
    raise SystemExit(1)

DST.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(SRC):
    root_path = Path(root)
    rel = root_path.relative_to(SRC)
    # prune excluded dirs
    dirs[:] = [d for d in dirs if d not in EXCLUDE_NAMES and not d.endswith('.egg-info')]
    for f in files:
        srcf = root_path / f
        relf = rel / f
        dstf = DST / relf
        try:
            if should_exclude(srcf):
                excluded.append(str(relf))
                continue
            dstf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcf, dstf)
            copied.append(str(relf))
            # preserve mode bits
            try:
                st = srcf.stat()
                dstf.chmod(stat.S_IMODE(st.st_mode))
            except Exception:
                pass
        except Exception as e:
            errors.append({'file': str(srcf), 'error': str(e)})

# Build .env.example from existing backend/.env if present (only keys)
env_example_path = DST / '.env.example'
src_env = SRC / '.env'
if src_env.exists():
    keys = []
    with src_env.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k = line.split('=', 1)[0].strip()
                keys.append(k)
    with env_example_path.open('w', encoding='utf-8') as out:
        out.write('# Generated .env.example (values redacted). Copy to .env and fill values.\n')
        for k in sorted(set(keys)):
            out.write(f"{k}=\n")
else:
    # write a recommended skeleton
    with env_example_path.open('w', encoding='utf-8') as out:
        out.write('# Example .env (fill values).\n')
        out.write('MONGO_URL=\n')
        out.write('DATABASE_NAME=whatsapp_agent_v2\n')
        out.write('WHATSAPP_ACCESS_TOKEN=\n')
        out.write('WHATSAPP_PHONE_NUMBER_ID=\n')
        out.write('WEBHOOK_VERIFY_TOKEN=whatsapp_webhook_2025\n')
        out.write('DISABLE_WHATSAPP_SENDS=true\n')
        out.write('GEMINI_API_KEY=\n')
        out.write('OPENROUTER_API_KEY=\n')

# copy run_with_ngrok.py from agent path if present
src_launcher = ROOT / 'agent' / 'whats-app-agent' / 'run_with_ngrok.py'
dst_launcher = Path(__file__).resolve().parent / 'run_with_ngrok.py'
if src_launcher.exists():
    try:
        shutil.copy2(src_launcher, dst_launcher)
        copied.append('run_with_ngrok.py')
    except Exception as e:
        errors.append({'file': str(src_launcher), 'error': str(e)})

report = {
    'moved_files': copied,
    'excluded_files': excluded,
    'modifications': [
        'Created .env.example (redacted).',
        'Preserved package layout under whats_app_agent/backend.'
    ],
    'run_instructions': [
        'cd whats_app_agent',
        'python -m venv .venv',
        '& .venv\\Scripts\\Activate.ps1',
        'python -m pip install -r backend/requirements.txt',
        'copy backend\\.env.example backend\\.env and fill secrets (do not paste here).',
        'place ngrok.exe into whats_app_agent root or ensure ngrok is on PATH',
        'python run_with_ngrok.py'
    ],
    'smoke_test_results': {},
    'errors': errors,
}

with (Path(__file__).resolve().parent / 'report.json').open('w', encoding='utf-8') as out:
    json.dump(report, out, indent=2)

print('Copy complete. report.json written to whats_app_agent/report.json')
if errors:
    print('Some errors occurred; see report.json')
