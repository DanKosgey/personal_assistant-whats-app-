"""Key management system that supports JSON configuration with metadata and usage tracking."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import os

logger = logging.getLogger(__name__)

class APIKey:
    def __init__(self, key: str, settings: Dict[str, Any]):
        self.key = key
        self.description = settings.get("description", "")
        self.quota_per_minute = settings.get("quota_per_minute", 60)
        self.enabled = settings.get("enabled", True)
        self.last_used: Optional[datetime] = None
        self._uses_this_minute = 0
        self._minute_start: Optional[datetime] = None

    def can_use(self) -> bool:
        """Check if key can be used based on quota."""
        now = datetime.utcnow()
        
        # Reset counter if we're in a new minute
        if not self._minute_start or (now - self._minute_start).total_seconds() >= 60:
            self._uses_this_minute = 0
            self._minute_start = now
        
        return self.enabled and self._uses_this_minute < self.quota_per_minute

    def record_use(self):
        """Record usage of this key."""
        now = datetime.utcnow()
        self.last_used = now
        
        if not self._minute_start or (now - self._minute_start).total_seconds() >= 60:
            self._uses_this_minute = 0
            self._minute_start = now
        
        self._uses_this_minute += 1

class KeyManager:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        # Ensure config_path is always a string (avoid None for open())
        self.config_path = str(config_path) if config_path else os.getenv("API_KEYS_CONFIG", "")
        if not self.config_path:
            self.config_path = str(Path(__file__).parent.parent / "config" / "api_keys.json")
        
        self.gemini_keys: List[APIKey] = []
        self.openrouter_keys: List[APIKey] = []
        self.gemini_settings: Dict[str, Any] = {}
        self.openrouter_settings: Dict[str, Any] = {}
        
        self._gemini_idx = 0
        self._openrouter_idx = 0
        self._next_retry: Optional[datetime] = None
        
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                config = json.load(f)

            # New simplified format expected:
            # {
            #   "keys": { "gemini": ["key1", "key2"], "openrouter": ["..."] },
            #   "settings": { "gemini": {...}, "openrouter": {...} }
            # }

            keys_root = config.get("keys", {})
            settings_root = config.get("settings", {})

            # Provider-level settings (may be empty)
            self.gemini_settings = settings_root.get("gemini", {})
            self.openrouter_settings = settings_root.get("openrouter", {})

            # Load Gemini keys (plain strings)
            self.gemini_keys = []
            gemini_list = keys_root.get("gemini", [])
            gemini_default_quota = self.gemini_settings.get("quota_per_minute", self.gemini_settings.get("default_quota_per_minute", 60))
            for k in gemini_list:
                # Each key is a plain string in the simplified format
                settings = {"quota_per_minute": gemini_default_quota, "enabled": True}
                self.gemini_keys.append(APIKey(key=k, settings=settings))

            # Load OpenRouter keys
            self.openrouter_keys = []
            openrouter_list = keys_root.get("openrouter", [])
            openrouter_default_quota = self.openrouter_settings.get("quota_per_minute", self.openrouter_settings.get("default_quota_per_minute", 50))
            for k in openrouter_list:
                settings = {"quota_per_minute": openrouter_default_quota, "enabled": True}
                self.openrouter_keys.append(APIKey(key=k, settings=settings))

            logger.info("Loaded %d Gemini keys and %d OpenRouter keys", len(self.gemini_keys), len(self.openrouter_keys))
        
        except FileNotFoundError:
            logger.warning("No config file found at %s, using environment variables", self.config_path)
            self._load_from_env()
        except Exception as e:
            logger.error("Error loading config: %s", e)
            self._load_from_env()

    def _load_from_env(self):
        """Fallback to loading from environment variables."""
        gemini_raw = os.getenv("GEMINI_API_KEYS", "")
        for key in [k.strip() for k in gemini_raw.split(",") if k.strip()]:
            self.gemini_keys.append(APIKey(key=key, settings={"quota_per_minute": 60, "enabled": True}))
        
        oraw = os.getenv("OPENROUTER_API_KEYS", "")
        for key in [k.strip() for k in oraw.split(",") if k.strip()]:
            self.openrouter_keys.append(APIKey(key=key, settings={"quota_per_minute": 50, "enabled": True}))
        
        self.gemini_settings = {"rotation_strategy": "round_robin"}
        self.openrouter_settings = {
            "enabled": os.getenv("ENABLE_OPENROUTER_FALLBACK", "").lower() in ("1", "true", "yes")
        }

    def save_config(self):
        """Save current configuration back to JSON file."""
        # Persist only the simplified structure (lists of keys and provider settings).
        config = {
            "keys": {
                "gemini": [k.key for k in self.gemini_keys],
                "openrouter": [k.key for k in self.openrouter_keys]
            },
            "settings": {
                "gemini": self.gemini_settings,
                "openrouter": self.openrouter_settings
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

        logger.info("Saved key configuration to %s", self.config_path)

    def next_gemini_key(self) -> Optional[APIKey]:
        """Get next available Gemini key."""
        if not self.gemini_keys:
            return None
        
        tried = 0
        while tried < len(self.gemini_keys):
            key = self.gemini_keys[self._gemini_idx % len(self.gemini_keys)]
            self._gemini_idx += 1
            
            if key.can_use():
                return key
            
            tried += 1
        
        return None

    def next_openrouter_key(self) -> Optional[APIKey]:
        """Get next available OpenRouter key."""
        if not self.openrouter_keys or not self.openrouter_settings.get("enabled", False):
            return None
        
        tried = 0
        while tried < len(self.openrouter_keys):
            key = self.openrouter_keys[self._openrouter_idx % len(self.openrouter_keys)]
            self._openrouter_idx += 1
            
            if key.can_use():
                return key
            
            tried += 1
        
        return None

    def record_use(self, key: APIKey):
        """Record usage of a key."""
        key.record_use()
    # Do not rewrite the simplified config on each usage (keeps file readable).
    # Usage is tracked in-memory; persist only when user explicitly calls save_config().

    def get_key_status(self) -> Dict[str, Any]:
        """Get current status of all keys."""
        return {
            "gemini": {
                "total_keys": len(self.gemini_keys),
                "available_keys": sum(1 for k in self.gemini_keys if k.can_use()),
                "keys": [
                    {
                        "description": k.description,
                        "enabled": k.enabled,
                        "uses_this_minute": k._uses_this_minute,
                        "quota_per_minute": k.quota_per_minute,
                        "last_used": k.last_used.isoformat() if k.last_used else None
                    }
                    for k in self.gemini_keys
                ]
            },
            "openrouter": {
                "enabled": self.openrouter_settings.get("enabled", False),
                "total_keys": len(self.openrouter_keys),
                "available_keys": sum(1 for k in self.openrouter_keys if k.can_use()),
                "keys": [
                    {
                        "description": k.description,
                        "enabled": k.enabled,
                        "uses_this_minute": k._uses_this_minute,
                        "quota_per_minute": k.quota_per_minute,
                        "last_used": k.last_used.isoformat() if k.last_used else None
                    }
                    for k in self.openrouter_keys
                ]
            }
        }
