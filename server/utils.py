import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def redact_secret(s: str) -> str:
    if not s:
        return s
    return s[:4] + "..." + s[-4:]


def get_logger(name: str):
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # default handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
