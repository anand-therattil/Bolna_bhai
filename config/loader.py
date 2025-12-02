import yaml
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent 
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)
