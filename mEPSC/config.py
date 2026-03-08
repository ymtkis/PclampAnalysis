from pathlib import Path
import yaml

def load_config(config_path=None, filename="config.yaml", max_up=6):

    if config_path is not None:
        p = Path(config_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    here = Path(__file__).resolve().parent
    for i, d in enumerate([here] + list(here.parents)):
        if i > max_up:
            break
        cand = d / filename
        if cand.exists():
            print(f"[config] using: {cand}")
            with open(cand, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Could not find {filename} from {here} up to {max_up} levels."
    )

CFG = load_config()
