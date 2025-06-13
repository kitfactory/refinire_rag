from typing import Dict, Any, Optional

class PluginConfig:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

class ConfigManager:
    def __init__(self):
        self.configs: Dict[str, PluginConfig] = {}

    def add_config(self, config: PluginConfig) -> None:
        self.configs[config.name] = config

    def get_config(self, name: str) -> Optional[PluginConfig]:
        return self.configs.get(name)

    def remove_config(self, name: str) -> None:
        if name in self.configs:
            del self.configs[name] 