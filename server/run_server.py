#!/usr/bin/env python3
"""
Simple server runner script that handles module imports correctly.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
os.environ.setdefault("PYTHONPATH", str(project_root))

if __name__ == "__main__":
    try:
        from server.server import app
        import uvicorn
        
        # Try to use uvloop for better performance if available
        try:
            import uvloop
            uvloop.install()
            loop = "uvloop"
        except ImportError:
            loop = "auto"
        
        print("🚀 Starting WhatsApp Assistant Server...")
        print(f"📁 Project root: {project_root}")
        print(f"🔄 Event loop: {loop}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            workers=1,  # Use single worker for development
            proxy_headers=True,
            forwarded_allow_ips="*",
            access_log=True,
            loop=loop,
            reload=False  # Disable auto-reload to prevent issues
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)