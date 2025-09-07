import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load .env files in a safe order, with root .env having priority
# Priority: repo root .env, agent/.env, agent/backend/.env
_PARENTS = Path(__file__).resolve().parents
# Prefer the directory one level above the package (repo root), guard against shallow paths
ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _PARENTS[0]
AGENT_DIR = Path(__file__).parents[1]

# First clear any existing values from previous loads
for key in ["GEMINI_API_KEYS", "OPENROUTER_API_KEYS"]:
    if key in os.environ:
        del os.environ[key]

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
            if not self.GEMINI_API_KEY:
                errors.append("GEMINI_API_KEY must be set")
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
