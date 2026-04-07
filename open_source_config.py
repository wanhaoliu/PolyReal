import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROOT_DIR / "PolyReal.json"
DEFAULT_REF_DIR = ROOT_DIR / "ref"
DEFAULT_RESULT_DIR = ROOT_DIR / "result"
DEFAULT_LOG_DIR = ROOT_DIR / "logs"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_api_config(base_url_env: str, api_key_env: str) -> tuple[str, str]:
    base_url = require_env(base_url_env).rstrip("/")
    api_key = require_env(api_key_env)
    return base_url, api_key


def build_headers(api_key: str) -> dict:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "PolyReal/1.0",
        "Content-Type": "application/json",
    }
