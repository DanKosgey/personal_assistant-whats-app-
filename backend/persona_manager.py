import os
import json
import glob
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Persona:
    def __init__(self, name: str, description: str, system_prompt: str):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt


class PersonaManager:
    """Simple persona manager that loads persona JSON files from a directory
    and exposes a runtime-selectable current persona. A persona JSON file
    should look like:

    {
      "name": "customer_support",
      "description": "Customer support persona for Acme Co.",
      "system_prompt": "You are Acme Co. support agent..."
    }

    The manager falls back to the provided default_prompt when no persona is
    selected.
    """

    def __init__(self, personas_dir: Optional[str] = None, default_prompt: Optional[str] = None):
        # Default personas directory next to this file unless overridden
        base_dir = os.path.dirname(__file__)
        self.personas_dir = personas_dir or os.getenv('PERSONAS_DIR') or os.path.join(base_dir, 'personas')
        self.default_prompt = default_prompt or "You are a helpful assistant."
        self._personas: Dict[str, Persona] = {}
        self.current: Optional[str] = None
        self.load_personas()

        # If an env var requests a persona, attempt to select it
        env_persona = os.getenv('AGENT_PERSONA')
        if env_persona:
            try:
                self.select_persona(env_persona)
            except Exception:
                logger.debug(f"Requested AGENT_PERSONA={env_persona} not found")

    def load_personas(self):
        """Load all persona JSON files from the personas directory."""
        self._personas = {}
        try:
            pattern = os.path.join(self.personas_dir, '*.json')
            for path in glob.glob(pattern):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        name = data.get('name') or os.path.splitext(os.path.basename(path))[0]
                        desc = data.get('description', '')
                        prompt = data.get('system_prompt', '')
                        if prompt:
                            self._personas[name] = Persona(name, desc, prompt)
                except Exception as e:
                    logger.warning(f"Failed to load persona file {path}: {e}")
        except Exception as e:
            logger.debug(f"No personas directory or load failed: {e}")

    def list_personas(self) -> List[Dict[str, str]]:
        return [{"name": p.name, "description": p.description} for p in self._personas.values()]

    def get_persona(self, name: str) -> Optional[Persona]:
        return self._personas.get(name)

    def select_persona(self, name: str) -> bool:
        if name not in self._personas:
            raise KeyError(f"Persona '{name}' not found")
        self.current = name
        logger.info(f"Persona selected: {name}")
        return True

    def add_persona(self, name: str, description: str, system_prompt: str) -> bool:
        # Save persona to file for persistence
        try:
            os.makedirs(self.personas_dir, exist_ok=True)
            path = os.path.join(self.personas_dir, f"{name}.json")
            payload = {"name": name, "description": description, "system_prompt": system_prompt}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            # Reload
            self.load_personas()
            return True
        except Exception as e:
            logger.error(f"Failed to add persona {name}: {e}")
            return False

    def get_system_prompt(self) -> str:
        if self.current:
            p = self._personas.get(self.current)
            if p:
                return p.system_prompt
        # Fall back to default prompt
        return self.default_prompt
