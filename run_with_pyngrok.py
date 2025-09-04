from pyngrok import ngrok
import time
import os
import sys
import subprocess
from pathlib import Path

# Get port from environment
PORT = int(os.environ.get("PORT", "8000"))

def start_server():
    """Start the FastAPI server"""
    ROOT = Path(__file__).resolve().parent
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = str(ROOT)

    cmd = [sys.executable, "-m", "uvicorn", "server.server:app", "--host", "127.0.0.1", "--port", str(PORT)]
    print("Starting uvicorn:", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)

def main():
    # Start the FastAPI server
    server_process = start_server()
    
    try:
        # Start ngrok tunnel
        print(f"Starting ngrok tunnel on port {PORT}...")
        public_url = ngrok.connect(str(PORT))
        print(f"Public URL: {public_url}")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up
        print("Closing ngrok tunnel...")
        ngrok.kill()
        
        print("Stopping server...")
        if server_process:
            server_process.terminate()

if __name__ == "__main__":
    main()
