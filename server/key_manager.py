from typing import Optional, Dict, Any, List, Union
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
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
        self._temporarily_unavailable = False
        self._unavailable_until: Optional[datetime] = None

    def can_use(self) -> bool:
        """Check if key can be used based on quota and availability."""
        now = datetime.now().astimezone()
        
        # Check if key is temporarily unavailable due to rate limiting
        if self._temporarily_unavailable and self._unavailable_until:
            if now < self._unavailable_until:
                logger.debug("Key %s is temporarily unavailable until %s", 
                           self.key[:8] if self.key else "None", 
                           self._unavailable_until.isoformat())
                return False
            else:
                # Reset availability
                logger.info("Key %s is now available again (was unavailable until %s)", 
                          self.key[:8] if self.key else "None", 
                          self._unavailable_until.isoformat())
                self._temporarily_unavailable = False
                self._unavailable_until = None
        
        # Reset counter if we're in a new minute
        if not self._minute_start or (now - self._minute_start).total_seconds() >= 60:
            if self._minute_start:
                logger.debug("Resetting usage counter for key %s (minute started at %s)", 
                           self.key[:8] if self.key else "None", 
                           self._minute_start.isoformat())
            self._uses_this_minute = 0
            self._minute_start = now
        
        can_use = self.enabled and self._uses_this_minute < self.quota_per_minute
        logger.debug("Key %s can_use check: enabled=%s, uses_this_minute=%d, quota=%d, result=%s", 
                   self.key[:8] if self.key else "None", 
                   self.enabled, self._uses_this_minute, self.quota_per_minute, can_use)
        return can_use

    def record_use(self):
        """Record usage of this key."""
        now = datetime.now().astimezone()
        self.last_used = now
        
        if not self._minute_start or (now - self._minute_start).total_seconds() >= 60:
            self._uses_this_minute = 0
            self._minute_start = now
        
        self._uses_this_minute += 1
        logger.debug("Recorded use for key %s: uses_this_minute=%d, minute_start=%s", 
                   self.key[:8] if self.key else "None", 
                   self._uses_this_minute, 
                   self._minute_start.isoformat() if self._minute_start else "None")

    def mark_unavailable(self, seconds: int = 60):
        """Mark key as temporarily unavailable due to rate limiting."""
        self._temporarily_unavailable = True
        self._unavailable_until = datetime.now().astimezone() + timedelta(seconds=seconds)
        logger.warning("Key %s marked as temporarily unavailable for %d seconds (until %s)", 
                     self.key[:8] if self.key else "None", 
                     seconds, self._unavailable_until.isoformat())

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
            
            # If no keys are configured, log a warning but don't fail
            if len(self.gemini_keys) == 0 and len(self.openrouter_keys) == 0:
                logger.warning("No API keys configured. AI functionality will be disabled until keys are added.")
        
        except FileNotFoundError:
            logger.warning("No config file found at %s, using environment variables", self.config_path)
            self._load_from_env()
        except Exception as e:
            logger.error("Error loading config: %s", e)
            self._load_from_env()

    def _load_from_env(self):
        """Fallback to loading from environment variables."""
        gemini_raw = os.getenv("GEMINI_API_KEYS", "")
        # Also check for single GEMINI_API_KEY
        single_gemini = os.getenv("GEMINI_API_KEY", "")
        
        # Combine both sources
        all_gemini_keys = []
        if single_gemini:
            all_gemini_keys.append(single_gemini.strip())
        if gemini_raw:
            all_gemini_keys.extend([k.strip() for k in gemini_raw.split(",") if k.strip()])
            
        for key in all_gemini_keys:
            self.gemini_keys.append(APIKey(key=key, settings={"quota_per_minute": 60, "enabled": True}))
        
        oraw = os.getenv("OPENROUTER_API_KEYS", "")
        for key in [k.strip() for k in oraw.split(",") if k.strip()]:
            self.openrouter_keys.append(APIKey(key=key, settings={"quota_per_minute": 50, "enabled": True}))
        
        self.gemini_settings = {"rotation_strategy": "round_robin"}
        self.openrouter_settings = {
            "enabled": os.getenv("ENABLE_OPENROUTER_FALLBACK", "").lower() in ("1", "true", "yes")
        }
        
        # Log the number of keys loaded from environment
        logger.info("Loaded %d Gemini keys and %d OpenRouter keys from environment variables", 
                   len(self.gemini_keys), len(self.openrouter_keys))

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

        # Ensure the config directory exists
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

        logger.info("Saved key configuration to %s", self.config_path)

    def next_gemini_key(self) -> Optional[APIKey]:
        """Get next available Gemini key."""
        if not self.gemini_keys:
            logger.debug("No Gemini keys available")
            return None
        
        tried = 0
        while tried < len(self.gemini_keys):
            key = self.gemini_keys[self._gemini_idx % len(self.gemini_keys)]
            self._gemini_idx += 1
            
            logger.debug("Checking Gemini key %d/%d: %s", 
                        tried + 1, len(self.gemini_keys), 
                        key.key[:8] if key.key else "None")
            
            if key.can_use():
                logger.debug("Selected Gemini key %s for use", key.key[:8] if key.key else "None")
                return key
            
            tried += 1
        
        logger.warning("No available Gemini keys found after checking all %d keys", len(self.gemini_keys))
        return None

    def next_openrouter_key(self) -> Optional[APIKey]:
        """Get next available OpenRouter key."""
        if not self.openrouter_keys or not self.openrouter_settings.get("enabled", False):
            logger.debug("OpenRouter not enabled or no keys available")
            return None
        
        # If we have no keys, return None
        if len(self.openrouter_keys) == 0:
            logger.debug("No OpenRouter keys configured")
            return None
            
        tried = 0
        start_idx = self._openrouter_idx  # Remember where we started
        
        while tried < len(self.openrouter_keys):
            # Use modulo to ensure we cycle through all keys
            key = self.openrouter_keys[self._openrouter_idx % len(self.openrouter_keys)]
            self._openrouter_idx += 1
            
            logger.debug("Checking OpenRouter key %d/%d (index %d): %s", 
                        tried + 1, len(self.openrouter_keys), 
                        (self._openrouter_idx - 1) % len(self.openrouter_keys),
                        key.key[:8] if key.key else "None")
            
            if key.can_use():
                logger.debug("Selected OpenRouter key %s for use", key.key[:8] if key.key else "None")
                return key
            
            tried += 1
            
            # If we've tried all keys and none are available, break
            # But only break if we've actually tried all keys (not just when we loop back to start)
            if tried >= len(self.openrouter_keys):
                break
        
        logger.warning("No available OpenRouter keys found after checking all %d keys", len(self.openrouter_keys))
        return None

    def record_use(self, key: APIKey):
        """Record usage of a key."""
        key.record_use()
        logger.debug("Recorded usage for key %s", key.key[:8] if key.key else "None")
        
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

    def add_gemini_key(self, key: str, quota_per_minute: int = 60):
        """Add a new Gemini key."""
        settings = {"quota_per_minute": quota_per_minute, "enabled": True}
        self.gemini_keys.append(APIKey(key=key, settings=settings))
        self.save_config()

    def add_openrouter_key(self, key: str, quota_per_minute: int = 50):
        """Add a new OpenRouter key."""
        settings = {"quota_per_minute": quota_per_minute, "enabled": True}
        self.openrouter_keys.append(APIKey(key=key, settings=settings))
        self.save_config()

    def remove_gemini_key(self, key: str):
        """Remove a Gemini key."""
        self.gemini_keys = [k for k in self.gemini_keys if k.key != key]
        self.save_config()

    def remove_openrouter_key(self, key: str):
        """Remove an OpenRouter key."""
        self.openrouter_keys = [k for k in self.openrouter_keys if k.key != key]
        self.save_config()

    def mark_gemini_key_unavailable(self, key_str: str, seconds: int = 60):
        """Mark a specific Gemini key as temporarily unavailable."""
        for key in self.gemini_keys:
            if key.key == key_str:
                key.mark_unavailable(seconds)
                break

    def mark_openrouter_key_unavailable(self, key_str: str, seconds: int = 60):
        """Mark a specific OpenRouter key as temporarily unavailable."""
        for key in self.openrouter_keys:
            if key.key == key_str:
                key.mark_unavailable(seconds)
                break