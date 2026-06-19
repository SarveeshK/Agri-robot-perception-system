import yaml
import os

class Config:
    """
    Singleton-like configuration loader.
    """
    _instance = None
    _config_data = {}

    def __new__(cls, config_path="config/settings.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as f:
            self._config_data = yaml.safe_load(f)

    @property
    def camera(self):
        return self._config_data.get("camera", {})

    @property
    def yolo(self):
        return self._config_data.get("yolo", {})

    @property
    def measurement(self):
        return self._config_data.get("measurement", {})

    @property
    def obstacle(self):
        return self._config_data.get("obstacle", {})

    @property
    def visualization(self):
        return self._config_data.get("visualization", {})

    @property
    def logging(self):
        return self._config_data.get("logging", {})

# Global instance for easy access
settings = Config()
