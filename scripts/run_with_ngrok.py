"""Production-ready launcher for FastAPI server with ngrok integration.

Features:
- Robust error handling and logging
- Configuration management
- Health checks and monitoring
- Graceful shutdown handling
- Process management with proper cleanup
- Retry mechanisms and timeouts
- Production-safe defaults
"""

import os
import sys
import time
import signal
import subprocess
import requests
import logging
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager
import atexit


@dataclass
class Config:
    """Configuration class with validation and defaults."""
    port: str = "8000"
    host: str = "127.0.0.1"
    ngrok_timeout: int = 60
    health_check_timeout: int = 60  # Increased from 30 to 60 seconds
    max_retries: int = 3
    retry_delay: int = 5
    log_level: str = "DEBUG"
    workspace_root: Path = Path(__file__).resolve().parent.parent  # Changed from .parent to .parent.parent to get project root
    disable_db: bool = True
    dev_mode: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not (1 <= int(self.port) <= 65535):
            raise ValueError(f"Invalid port: {self.port}")
        
        if self.ngrok_timeout <= 0:
            raise ValueError("ngrok_timeout must be positive")
            
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        # Get the project root directory (parent of scripts directory)
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent  # Go up one level to get project root
        
        return cls(
            port=os.getenv("PORT", "8000"),
            host=os.getenv("HOST", "127.0.0.1"),
            ngrok_timeout=int(os.getenv("NGROK_TIMEOUT", "60")),
            health_check_timeout=int(os.getenv("HEALTH_CHECK_TIMEOUT", "60")),  # Increased from 30 to 60
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY", "5")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            workspace_root=project_root,  # Use project root instead of script directory
            disable_db=os.getenv("DISABLE_DB", "1").lower() in ("1", "true", "yes"),
            dev_mode=os.getenv("DEV_MODE", "1").lower() in ("1", "true", "yes"),
        )


class ProcessManager:
    """Manages child processes with proper cleanup."""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.logger = logging.getLogger(f"{__name__}.ProcessManager")
        self._shutdown_event = threading.Event()
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
        
        # Handle signals - with more defensive approach
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError) as e:
            self.logger.warning(f"Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_event.set()
        self.cleanup_all()
        # Don't exit immediately, let the main loop handle the shutdown
        # sys.exit(0)
    
    def add_process(self, process: subprocess.Popen, name: Optional[str] = None) -> subprocess.Popen:
        """Add a process to be managed."""
        self.processes.append(process)
        self.logger.info(f"Added process {name or 'unnamed'} with PID {process.pid}")
        return process
    
    def cleanup_all(self):
        """Clean up all managed processes."""
        if not self.processes:
            return
            
        self.logger.info("Cleaning up processes...")
        for process in self.processes[:]:  # Copy list to avoid modification during iteration
            self._cleanup_process(process)
        self.processes.clear()
    
    def _cleanup_process(self, process: subprocess.Popen, timeout: int = 10):
        """Clean up a single process gracefully."""
        if process.poll() is not None:
            return  # Already terminated
        
        try:
            # Try graceful termination first
            process.terminate()
            process.wait(timeout=timeout)
            self.logger.info(f"Process {process.pid} terminated gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            try:
                process.kill()
                process.wait(timeout=5)
                self.logger.warning(f"Process {process.pid} force killed")
            except Exception as e:
                self.logger.error(f"Failed to kill process {process.pid}: {e}")
        except Exception as e:
            self.logger.error(f"Error cleaning up process {process.pid}: {e}")


class LauncherError(Exception):
    """Custom exception for launcher errors."""
    pass


class ProductionLauncher:
    """Production-ready launcher with comprehensive error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.process_manager = ProcessManager()
        self.logger = self._setup_logging()
        self._load_environment()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Remove existing handlers to avoid duplication
        logger.handlers.clear()
        
        # Console handler with formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        from dotenv import load_dotenv
        
        env_file = self.config.workspace_root / '.env'
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            self.logger.info(f"Loaded environment from {env_file}")
        else:
            self.logger.info("No .env file found, using system environment")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise
                
                delay = self.config.retry_delay * (2 ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    
    def cleanup_existing_ngrok(self):
        """Clean up existing ngrok processes to avoid conflicts."""
        self.logger.info("Cleaning up existing ngrok processes...")
        
        # Use a simple and safe approach with system commands only
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ['taskkill', '/F', '/IM', 'ngrok.exe'], 
                    capture_output=True, 
                    check=False, 
                    timeout=10
                )
                if result.returncode == 0:
                    self.logger.info("Successfully killed existing ngrok processes")
                else:
                    # Non-zero return code is normal if no ngrok processes exist
                    self.logger.debug("No ngrok processes found to kill")
            else:  # Unix-like
                result = subprocess.run(
                    ['pkill', '-f', 'ngrok'], 
                    capture_output=True, 
                    check=False, 
                    timeout=10
                )
                self.logger.debug(f"pkill result: {result.returncode}")
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout while killing existing ngrok processes")
        except Exception as e:
            self.logger.warning(f"Failed to kill existing ngrok processes: {e}")
        
        self.logger.debug("Ngrok cleanup completed")
    
    def _start_monitoring_thread(self):
        """Start a monitoring thread to show server activity"""
        def monitor_server():
            """Monitor server health and show activity"""
            last_check = time.time()
            while True:
                try:
                    # Check every 30 seconds
                    time.sleep(30)
                    
                    # Get current time for logging
                    current_time = time.strftime("%H:%M:%S")
                    
                    # Health check
                    health_url = f"http://{self.config.host}:{self.config.port}/health"
                    try:
                        response = requests.get(health_url, timeout=5)
                        if response.status_code == 200:
                            self.logger.info(f"[{current_time}] 👨‍⚕️ Server is healthy and running")
                        else:
                            self.logger.warning(f"Health check returned status: {response.status_code}")
                    except requests.RequestException as e:
                        self.logger.error(f"Health check failed: {e}")
                        break
                    
                    # Show ngrok tunnel status
                    try:
                        ngrok_api = "http://127.0.0.1:4040/api/tunnels"
                        response = requests.get(ngrok_api, timeout=3)
                        if response.status_code == 200:
                            data = response.json()
                            tunnels = data.get("tunnels", [])
                            if tunnels:
                                tunnel = tunnels[0]
                                connections = tunnel.get("metrics", {}).get("conns", {}).get("count", 0)
                                if connections > 0:
                                    self.logger.info(f"[{current_time}] 📈 Tunnel active - {connections} connections")
                    except:
                        pass  # Ignore ngrok API errors
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.debug(f"Monitor thread error: {e}")
                    time.sleep(10)  # Wait longer on errors
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_server, daemon=True)
        monitor_thread.start()
        self.logger.debug("Started monitoring thread")
    
    def find_ngrok(self) -> Optional[str]:
        """Find ngrok executable in common locations."""
        candidates = [
            self.config.workspace_root / "ngrok",
            self.config.workspace_root / "ngrok.exe",
            self.config.workspace_root / "ngrok" / "ngrok",
            self.config.workspace_root / "ngrok" / "ngrok.exe",
            self.config.workspace_root / "tools" / "ngrok",
            self.config.workspace_root / "tools" / "ngrok.exe",
        ]
        
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                # Verify it's executable
                if os.access(candidate, os.X_OK):
                    self.logger.info(f"Found ngrok executable: {candidate}")
                    return str(candidate)
                else:
                    self.logger.warning(f"Found ngrok file but it's not executable: {candidate}")
        
        # Check system PATH
        import shutil
        system_ngrok = shutil.which('ngrok')
        if system_ngrok:
            self.logger.info(f"Found ngrok in system PATH: {system_ngrok}")
            return system_ngrok
        
        self.logger.warning("No ngrok executable found")
        return None
    
    def start_uvicorn(self) -> subprocess.Popen:
        """Start uvicorn server with proper configuration."""
        self.logger.info("Starting uvicorn server...")
        
        # Prepare environment
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = str(self.config.workspace_root) + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = str(self.config.workspace_root)
        
        # Set production-safe defaults
        env.setdefault("DISABLE_DB", "1" if self.config.disable_db else "0")
        env.setdefault("DEV_SMOKE", "0")
        
        self.logger.debug(f"Workspace root: {self.config.workspace_root}")
        self.logger.debug(f"Python path: {env.get('PYTHONPATH')}")
        
        # Try different Python executables
        python_candidates = [
            sys.executable,  # Current Python executable (most reliable)
            "python3",
            "python",
        ]
        
        for python_exe in python_candidates:
            try:
                self.logger.debug(f"Trying Python executable: {python_exe}")
                # Test if uvicorn is available
                test_cmd = [python_exe, "-c", "import uvicorn; print('OK')"]
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                
                self.logger.debug(f"Test result: returncode={result.returncode}, stdout='{result.stdout}', stderr='{result.stderr}'")
                
                if result.returncode == 0 and "OK" in result.stdout:
                    self.logger.info(f"Using Python executable: {python_exe}")
                    
                    cmd = [
                        python_exe, "-m", "uvicorn", "server.server:app",
                        "--host", self.config.host,
                        "--port", self.config.port,
                        "--log-level", "debug",
                        "--access-log",
                        "--reload" if self.config.dev_mode else "--no-reload"
                    ]
                    
                    # Remove conflicting options
                    if "--no-access-log" in cmd:
                        cmd.remove("--no-access-log")
                    
                    self.logger.info(f"Starting uvicorn: {' '.join(cmd)}")
                    self.logger.debug(f"Working directory: {self.config.workspace_root}")
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(self.config.workspace_root),
                        env=env,
                        stdout=None,  # Show stdout in real-time
                        stderr=None,  # Show stderr in real-time
                    )
                    
                    self.logger.info(f"Started uvicorn process with PID: {process.pid}")
                    
                    # Check if the process started successfully
                    time.sleep(1)  # Give the process a moment to start
                    if process.poll() is not None:
                        self.logger.error(f"Uvicorn process exited immediately with code: {process.returncode}")
                        raise LauncherError("Uvicorn process failed to start")
                    
                    return self.process_manager.add_process(process, "uvicorn")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
                self.logger.warning(f"Failed to start with {python_exe}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error with {python_exe}: {e}", exc_info=True)
                continue
        
        raise LauncherError("Could not find a Python executable with uvicorn installed")
    
    def start_ngrok(self, ngrok_path: str) -> subprocess.Popen:
        """Start ngrok tunnel with configuration."""
        self.logger.info("Starting ngrok tunnel...")
        
        # Check for config file first
        config_file = self.config.workspace_root / "ngrok.yml"
        
        if config_file.exists():
            cmd = [ngrok_path, "start", "whatsapp-webhook", "--config", str(config_file)]
            self.logger.info(f"Using ngrok config file: {config_file}")
        else:
            cmd = [ngrok_path, "http", f"{self.config.host}:{self.config.port}"]
            self.logger.info("Using default ngrok configuration")
        
        self.logger.info(f"Starting ngrok: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            cwd=str(self.config.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        return self.process_manager.add_process(process, "ngrok")
    
    def wait_for_server_health(self) -> bool:
        """Wait for the server to be healthy."""
        self.logger.info("Waiting for server to be healthy...")
        
        health_url = f"http://{self.config.host}:{self.config.port}/health"
        deadline = time.time() + self.config.health_check_timeout
        
        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    self.logger.info("Server is healthy")
                    return True
                else:
                    self.logger.debug(f"Health check returned status {response.status_code}")
            except requests.RequestException as e:
                self.logger.debug(f"Health check request failed: {e}")
            except Exception as e:
                self.logger.debug(f"Unexpected error during health check: {e}")
            
            time.sleep(1)
        
        self.logger.error("Server health check failed - server did not become healthy within timeout")
        self.logger.error(f"Check if the server is running at {health_url}")
        # Add more diagnostic information
        self.logger.error("This could be due to:")
        self.logger.error("1. Server failed to start (check logs above)")
        self.logger.error("2. Database connection issues")
        self.logger.error("3. Port already in use")
        self.logger.error("4. Missing dependencies")
        self.logger.error("5. Configuration issues")
        return False
    
    def wait_for_ngrok_url(self) -> Optional[str]:
        """Wait for ngrok to provide a public URL."""
        self.logger.info("Waiting for ngrok public URL...")
        
        api_url = "http://127.0.0.1:4040/api/tunnels"
        deadline = time.time() + self.config.ngrok_timeout
        
        while time.time() < deadline:
            try:
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    tunnels = data.get("tunnels", [])
                    
                    for tunnel in tunnels:
                        public_url = tunnel.get("public_url")
                        if public_url and public_url.startswith("https"):
                            self.logger.info(f"Got ngrok public URL: {public_url}")
                            return public_url
                            
            except requests.RequestException as e:
                self.logger.debug(f"ngrok API request failed: {e}")
            
            time.sleep(1)
        
        self.logger.error("Failed to get ngrok public URL within timeout")
        return None
    
    def start_pyngrok_fallback(self) -> Optional[str]:
        """Fallback to pyngrok if native ngrok fails."""
        try:
            from pyngrok import ngrok as pyngrok
            from pyngrok.conf import PyngrokConfig
            
            self.logger.info("Starting pyngrok fallback...")
            
            # Configure pyngrok
            config = PyngrokConfig(api_key=os.getenv("NGROK_API_KEY"))
            
            tunnel = pyngrok.connect(
                f"{self.config.host}:{self.config.port}",
                pyngrok_config=config
            )
            
            public_url = str(tunnel.public_url)
            if not public_url.startswith("https"):
                public_url = public_url.replace("http", "https", 1)
            
            self.logger.info(f"pyngrok tunnel established: {public_url}")
            return public_url
            
        except ImportError:
            self.logger.error("pyngrok not available for fallback")
            return None
        except Exception as e:
            self.logger.error(f"pyngrok fallback failed: {e}")
            return None
    
    def register_webhook(self, public_url: str) -> bool:
        """Register webhook using the existing script."""
        self.logger.info("Registering webhook...")
        
        script_path = self.config.workspace_root / "backend" / "check_and_register_webhook.py"
        if not script_path.exists():
            self.logger.error(f"Webhook registration script not found: {script_path}")
            return False
        
        env = os.environ.copy()
        env["NGROK_PUBLIC_URL"] = public_url
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                env=env,
                timeout=30,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Webhook registered successfully")
                return True
            else:
                self.logger.error(f"Webhook registration failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Webhook registration timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error during webhook registration: {e}")
            return False
    
    def run(self) -> int:
        """Main run method with comprehensive error handling."""
        try:
            self.logger.info("Starting production launcher...")
            
            # Clean up existing processes
            self.cleanup_existing_ngrok()
            
            # Start uvicorn server
            self.logger.info("Attempting to start uvicorn server...")
            uvicorn_process = self._retry_with_backoff(self.start_uvicorn)
            self.logger.info("Uvicorn server started successfully")
            
            # Wait for server to be healthy
            self.logger.info("Waiting for server health check...")
            if not self.wait_for_server_health():
                self.logger.error("Server failed health check. Checking server status...")
                # Try to get more information about why the server failed
                health_url = f"http://{self.config.host}:{self.config.port}/health"
                try:
                    response = requests.get(health_url, timeout=5)
                    self.logger.error(f"Health endpoint returned status {response.status_code}: {response.text}")
                except requests.RequestException as e:
                    self.logger.error(f"Could not reach health endpoint: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error when checking health endpoint: {e}")
                
                raise LauncherError("Server failed health check")
            
            # Try to start ngrok
            public_url = None
            ngrok_path = self.find_ngrok()
            
            if ngrok_path:
                try:
                    self.logger.info("Attempting to start ngrok...")
                    ngrok_process = self.start_ngrok(ngrok_path)
                    public_url = self.wait_for_ngrok_url()
                    if public_url:
                        self.logger.info("Ngrok started successfully")
                    else:
                        self.logger.warning("Ngrok failed to provide public URL")
                except Exception as e:
                    self.logger.warning(f"Native ngrok failed: {e}")
            
            # Fallback to pyngrok if needed
            if not public_url:
                self.logger.info("Attempting pyngrok fallback...")
                public_url = self.start_pyngrok_fallback()
            
            # Register webhook if we have a public URL
            if public_url:
                self.logger.info("Attempting to register webhook...")
                webhook_success = self._retry_with_backoff(self.register_webhook, public_url)
                if not webhook_success:
                    self.logger.warning("Webhook registration failed, but continuing...")
            else:
                self.logger.warning("No public URL available, skipping webhook registration")
            
            self.logger.info("All services started successfully. Press Ctrl+C to stop.")
            
            # Add monitoring info
            self.logger.info(f"🌐 Server running at: http://{self.config.host}:{self.config.port}")
            if public_url:
                self.logger.info(f"🔗 Public URL: {public_url}")
                self.logger.info(f"📱 WhatsApp webhook: {public_url}/api/webhook")
                self.logger.info(f"🧪 Test webhook: curl -X POST {public_url}/api/webhook")
            self.logger.info(f"📊 Health check: http://{self.config.host}:{self.config.port}/health")
            self.logger.info(f"📋 API docs: http://{self.config.host}:{self.config.port}/api/docs")
            self.logger.info(f"🎭 Personas: http://{self.config.host}:{self.config.port}/api/personas")
            self.logger.info("")
            self.logger.info("🔍 MONITORING: Watching for incoming WhatsApp messages...")
            self.logger.info("📝 LOGS: All server activity will be shown below:")
            self.logger.info("" + "="*60)
            
            # Start monitoring thread
            self._start_monitoring_thread()
            
            # Wait for uvicorn to exit
            if uvicorn_process:
                self.logger.info("Waiting for uvicorn process to exit...")
                try:
                    # Wait for a short time to see if the process exits immediately
                    try:
                        exit_code = uvicorn_process.wait(timeout=1)
                        self.logger.info(f"Uvicorn process exited with code: {exit_code}")
                        return exit_code
                    except subprocess.TimeoutExpired:
                        # Process is still running, continue normally
                        self.logger.info("Uvicorn process is still running, waiting indefinitely...")
                        exit_code = uvicorn_process.wait()
                        self.logger.info(f"Uvicorn process exited with code: {exit_code}")
                        return exit_code
                except KeyboardInterrupt:
                    self.logger.info("Received keyboard interrupt, shutting down...")
                    return 0
                except Exception as e:
                    self.logger.error(f"Error waiting for uvicorn process: {e}", exc_info=True)
                    return 1
            else:
                self.logger.info("No uvicorn process to wait for, exiting...")
                return 0
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
            return 0
        except LauncherError as e:
            self.logger.error(f"Launcher error: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            return 1
        finally:
            self.process_manager.cleanup_all()


def main():
    """Entry point with configuration and error handling."""
    try:
        print("Starting WhatsApp AI Agent launcher...")
        config = Config.from_env()
        print(f"Configuration loaded: port={config.port}, workspace={config.workspace_root}")
        launcher = ProductionLauncher(config)
        print("Launcher initialized, starting services...")
        exit_code = launcher.run()
        print(f"Launcher exited with code: {exit_code}")
        return exit_code
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
        return 0
    except Exception as e:
        print(f"Failed to initialize launcher: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())