import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from src.utils.hf_utils import BASE_DIR, configure_hf_environment


MODEL_TARGETS = {
    "bge-m3": ("BAAI/bge-m3", BASE_DIR / "models" / "embedding" / "bge-m3"),
    "reranker": ("BAAI/bge-reranker-base", BASE_DIR / "models" / "reranker" / "bge-reranker-base"),
    "router": ("sentence-transformers/stsb-roberta-large", BASE_DIR / "models" / "router" / "stsb-roberta-large"),
}


def download_model(name: str) -> None:
    repo_id, target_dir = MODEL_TARGETS[name]
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        resume_download=True,
        local_dir_use_symlinks=False,
        max_workers=1,
    )
    print(f"Done: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download required Hugging Face models for ScholarRAG")
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_TARGETS.keys()],
        default="all",
        help="Which model to download",
    )
    args = parser.parse_args()

    configure_hf_environment()

    names = list(MODEL_TARGETS.keys()) if args.model == "all" else [args.model]
    for name in names:
        download_model(name)


if __name__ == "__main__":
    main()
