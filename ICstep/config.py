from pathlib import Path
import yaml

def load_config(config_path="config.yaml"):

    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

CFG = load_config()
