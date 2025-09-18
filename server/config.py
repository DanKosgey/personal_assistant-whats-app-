import os
from typing import Optional
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load .env files in a safe order, with root .env having priority
# Priority: repo root .env, agent/.env, agent/backend/.env
_PARENTS = Path(__file__).resolve().parents
# Prefer the directory one level above the package (repo root), guard against shallow paths
ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _PARENTS[0]
AGENT_DIR = Path(__file__).parents[1]

logger = logging.getLogger(__name__)

# Do not delete existing env vars; respect process-provided secrets

# Now load root .env first with override
try:
    load_dotenv(dotenv_path=ROOT / '.env', override=True)
except Exception as e:
    print(f"Error loading root .env: {e}")

# Then load others without override
try:
    load_dotenv(dotenv_path=AGENT_DIR / '.env', override=False)
except Exception:
    pass


class AgentConfig:
    # Application Settings
    APP_NAME: str = os.getenv("APP_NAME", "WhatsApp AI Agent")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes")
    ENV: str = os.getenv("ENV", "development")
    PORT: int = int(os.getenv("PORT", "8001"))
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")  # Required in production
    ALLOWED_HOSTS: list = os.getenv("ALLOWED_HOSTS", "*").split(",")
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    
    # WhatsApp API Settings
    WHATSAPP_ACCESS_TOKEN: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    WEBHOOK_VERIFY_TOKEN: str = os.getenv("WEBHOOK_VERIFY_TOKEN", "")
    DISABLE_WHATSAPP_SENDS: bool = os.getenv("DISABLE_WHATSAPP_SENDS", "False").lower() in (
        "1", "true", "yes"
    )
    
    # Database Settings
    MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "whatsapp_agent")
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    DISABLE_DB: bool = os.getenv("DISABLE_DB", "False").lower() in ("1", "true", "yes")
    
    # AI Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    AI_MODEL: str = os.getenv("AI_MODEL", "gemini-pro")
    
    # Feature Settings
    PERSONAS_DIR: str = os.getenv("PERSONAS_DIR", "personas")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "100"))  # Requests per minute
    # Development/testing helpers
    DEV_SMOKE: bool = os.getenv("DEV_SMOKE", "False").lower() in ("1", "true", "yes")
    
    # Gatekeeper Settings
    NOTIFY_THRESHOLD: int = int(os.getenv("NOTIFY_THRESHOLD", "40"))
    EMBED_SAVE_THRESHOLD: int = int(os.getenv("EMBED_SAVE_THRESHOLD", "25"))
    INACTIVITY_TIMEOUT_HOURS: int = int(os.getenv("INACTIVITY_TIMEOUT_HOURS", "4"))
    FOLLOWUP_PROMPTS_ALLOWED: int = int(os.getenv("FOLLOWUP_PROMPTS_ALLOWED", "2"))
    MEANINGFUL_CONVERSATION_THRESHOLD: float = float(os.getenv("MEANINGFUL_CONVERSATION_THRESHOLD", "0.5"))
    MIN_CONVERSATION_LENGTH: int = int(os.getenv("MIN_CONVERSATION_LENGTH", "2"))
    
    # EOC (End of Conversation) Settings
    EOC_CONFIDENCE_THRESHOLD: float = float(os.getenv("EOC_CONFIDENCE_THRESHOLD", "0.60"))
    EOC_TRANSITION_THRESHOLD: float = float(os.getenv("EOC_TRANSITION_THRESHOLD", "0.70"))
    
    # Owner Notification Thresholds
    OWNER_NOTIFICATION_THRESHOLD: float = float(os.getenv("OWNER_NOTIFICATION_THRESHOLD", "0.8"))
    HIGH_PRIORITY_THRESHOLD: float = float(os.getenv("HIGH_PRIORITY_THRESHOLD", "0.9"))
    MEDIUM_PRIORITY_THRESHOLD: float = float(os.getenv("MEDIUM_PRIORITY_THRESHOLD", "0.7"))
    LOW_PRIORITY_THRESHOLD: float = float(os.getenv("LOW_PRIORITY_THRESHOLD", "0.5"))
    VIP_NOTIFICATION_THRESHOLD: float = float(os.getenv("VIP_NOTIFICATION_THRESHOLD", "0.6"))
    URGENT_NOTIFICATION_THRESHOLD: float = float(os.getenv("URGENT_NOTIFICATION_THRESHOLD", "0.4"))
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "False").lower() in ("1", "true", "yes")
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")

    def validate_production(self) -> list[str]:
        """Validate required production settings"""
        errors = []
        if self.ENV == "production":
            if not self.SECRET_KEY:
                errors.append("SECRET_KEY must be set in production")
            if not self.WHATSAPP_ACCESS_TOKEN:
                errors.append("WHATSAPP_ACCESS_TOKEN must be set")
            if not self.WHATSAPP_PHONE_NUMBER_ID:
                errors.append("WHATSAPP_PHONE_NUMBER_ID must be set")
            if not self.WEBHOOK_VERIFY_TOKEN:
                errors.append("WEBHOOK_VERIFY_TOKEN must be set")
            # Accept either single GEMINI_API_KEY or any from get_ai_keys()
            try:
                from .config import get_ai_keys  # type: ignore
                keys = get_ai_keys()
            except Exception:
                keys = []
            if not (self.GEMINI_API_KEY or keys):
                errors.append("GEMINI_API_KEY or GEMINI_API_KEYS must be set")
            if "*" in self.ALLOWED_HOSTS:
                errors.append("ALLOWED_HOSTS should be configured in production")
        return errors


# Global config instance
config = AgentConfig()

# Validate production settings on import
if config.ENV == "production":
    errors = config.validate_production()
    if errors:
        raise ValueError(f"Production configuration errors: {', '.join(errors)}")

def get_ai_keys() -> list[str]:
    """Unified AI key accessor.

    Reads GEMINI_API_KEYS (comma-separated) and GEMINI_API_KEY (single),
    merges, de-duplicates preserving order.
    Also checks for keys in JSON config file as fallback.
    """
    keys: list[str] = []
    multi = os.getenv("GEMINI_API_KEYS", "")
    if multi:
        keys.extend([k.strip() for k in multi.split(",") if k.strip()])
    single = os.getenv("GEMINI_API_KEY", "")
    if single:
        keys.append(single.strip())
    
    # Fallback to JSON config file if no keys found in environment
    if not keys:
        try:
            config_path = Path(__file__).parent.parent / "config" / "api_keys.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    gemini_keys = config.get("gemini_keys", [])
                    keys.extend(gemini_keys)
                    if gemini_keys:
                        logging.info(f"Loaded {len(gemini_keys)} Gemini keys from JSON config")
        except Exception as e:
            logging.error(f"Error loading Gemini keys from JSON: {e}")
    
    out: list[str] = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    if not out:
        try:
            logger.warning("No AI keys configured")
        except Exception:
            pass
    return out
