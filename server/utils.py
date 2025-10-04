import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Build a minimal structured payload; avoid slow deep copies
        payload = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Common HTTP/access extras if provided
        for key in ("method", "path", "status_code", "client_addr", "user_agent"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        # Render as compact JSON without external deps
        try:
            import json

            return json.dumps(payload, separators=(",", ":"))
        except Exception:
            # Fallback to simple string if json fails
            return f"{payload['timestamp']} {payload['level']} {payload['logger']} - {payload['message']}"


def setup_logging() -> None:
    """Configure application and uvicorn loggers for stdout and optional rotation.

    Env vars:
      - LOG_LEVEL: DEBUG/INFO/WARN/ERROR (default INFO)
      - LOG_FORMAT: json|plain (default plain)
      - LOG_TO_FILE: 1/0 (default 0)
      - LOG_FILE: path (default /var/log/whatsapp-agent/app.log)
      - LOG_MAX_BYTES: rotate size bytes (default 10485760)
      - LOG_BACKUP_COUNT: files to keep (default 5)
    """

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "plain").lower()
    log_to_file = os.getenv("LOG_TO_FILE", "0").lower() in ("1", "true", "yes")
    log_file = os.getenv("LOG_FILE", "/var/log/whatsapp-agent/app.log")
    max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear existing handlers to avoid duplicates during reloads/tests
    for h in list(root.handlers):
        root.removeHandler(h)

    if log_format == "json":
        stream_formatter = JsonFormatter()
    else:
        stream_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    root.addHandler(stream_handler)

    if log_to_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(stream_formatter)
            root.addHandler(file_handler)
        except Exception:
            # Ignore file logging errors in container environments
            pass

    # Align uvicorn loggers with our configuration
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.propagate = True


def redact_secret(s: str) -> str:
    if not s:
        return s
    return s[:4] + "..." + s[-4:]


def get_logger(name: str) -> logging.Logger:
    """Return a logger; handlers are configured globally by setup_logging()."""
    return logging.getLogger(name)
