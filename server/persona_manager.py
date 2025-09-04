from pathlib import Path
from typing import Dict, Optional
import json


class PersonaManager:
    def __init__(self, personas_dir: str = "personas", default_prompt: Optional[str] = None):
        self.personas_dir = Path(personas_dir)
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        self.personas = {}
        self.current = None
        if default_prompt:
            self.current = {"name": "default", "system_prompt": default_prompt}
        self.load_personas()

    def load_personas(self):
        for p in self.personas_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.personas[data.get("name", p.stem)] = data
            except Exception:
                continue

    def list_personas(self):
        return list(self.personas.keys())

    def get_system_prompt(self):
        if self.current:
            return self.current.get("system_prompt")
        return None

    def select_persona(self, name: str) -> bool:
        if name in self.personas:
            self.current = self.personas[name]
            return True
        return False

    def add_persona(self, name: str, system_prompt: str, description: str = ""):
        data = {"name": name, "system_prompt": system_prompt, "description": description}
        p = self.personas_dir / f"{name}.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.personas[name] = data
