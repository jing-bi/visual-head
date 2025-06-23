import yaml
from pathlib import Path


class Strategy:
    _instance = None

    def __new__(cls, name=None):
        if cls._instance is None:
            cls._instance = super(Strategy, cls).__new__(cls)
            files = (Path(__file__).parent.parent / "exp" / "confs").glob("*.yaml")
            cls._instance.conf = {
                k: v for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True) for k, v in yaml.safe_load(open(file)).items()
            }.get(name)
            cls._instance.name = name

        return cls._instance

    def __getattr__(self, key):
        if key in self.conf:
            return self.conf[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "conf" or key == "_instance":
            super().__setattr__(key, value)
        else:
            self.conf[key] = value
