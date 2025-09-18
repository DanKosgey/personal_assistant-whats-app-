#!/usr/bin/env python3
"""
Simple deployment script for WhatsApp AI Agent
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    print(f"Starting WhatsApp AI Agent deployment...")
    print(f"Project root: {project_root}")
    
    # Set up environment
    env = os.environ.copy()
    
    # Run the main application
    try:
        print("Starting application with ngrok...")
        subprocess.run([
            sys.executable, "run_with_ngrok.py"
        ], cwd=project_root, env=env)
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())