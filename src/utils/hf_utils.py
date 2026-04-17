import os
from pathlib import Path
from typing import Optional, Union


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = BASE_DIR / "models"


def configure_hf_environment() -> None:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _is_ready_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    required_any = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.onnx",
        "config.json",
        "modules.json",
        "tokenizer.json",
    ]
    existing = {child.name for child in path.iterdir()}
    return "config.json" in existing and any(name in existing for name in required_any if name != "config.json")


def _find_cached_snapshot(repo_id: str, cache_root: Optional[Union[str, Path]] = None) -> Optional[Path]:
    root = Path(cache_root or os.environ.get("HF_HOME") or DEFAULT_HF_HOME)
    repo_dir = root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return None

    candidates = sorted([path for path in repo_dir.iterdir() if path.is_dir()], reverse=True)
    for candidate in candidates:
        if _is_ready_model_dir(candidate):
            return candidate
    return None


def resolve_model_source(repo_id: str, preferred_local_dir: Optional[Union[str, Path]] = None) -> str:
    configure_hf_environment()

    if preferred_local_dir is not None:
        preferred_path = Path(preferred_local_dir)
        if _is_ready_model_dir(preferred_path):
            return str(preferred_path)

    direct_path = Path(repo_id)
    if direct_path.exists() and _is_ready_model_dir(direct_path):
        return str(direct_path)

    cached_snapshot = _find_cached_snapshot(repo_id)
    if cached_snapshot is not None:
        return str(cached_snapshot)

    return repo_id
