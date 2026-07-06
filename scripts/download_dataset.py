"""
download_dataset.py
===================
Provider-agnostic dataset downloader dispatcher.

Reads config/dataset_sources.yaml to discover registered providers,
then dynamically loads the correct provider module and calls download().

Usage:
    python scripts/download_dataset.py --provider openimages --limit 200
    python scripts/download_dataset.py --provider openimages --target Tree Rock --limit 100
    python scripts/download_dataset.py --provider roboflow --url https://...
    python scripts/download_dataset.py --provider kaggle --dataset user/dataset-name
    python scripts/download_dataset.py --provider manual --zip /path/to/file.zip --name my_data

Adding a new provider:
    1. Create scripts/providers/<name>.py  with a download(args) function
    2. Add an entry to config/dataset_sources.yaml
    No changes needed in this file.
"""

import argparse
import importlib
import sys
import os
import yaml

# Add project root to sys.path so 'scripts.providers' can be resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


CONFIG_PATH = "config/dataset_sources.yaml"


def load_providers() -> dict:
    """Load provider registry from dataset_sources.yaml."""
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("providers", {})


def get_enabled_providers(providers: dict) -> dict:
    return {name: cfg for name, cfg in providers.items() if cfg.get("enabled", True)}


def build_parser(enabled: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AgriVision Dataset Downloader — provider-agnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name}: {cfg['notes']}" for name, cfg in enabled.items()
        ),
    )
    parser.add_argument(
        "--provider",
        choices=list(enabled.keys()),
        required=True,
        help="Dataset provider to use",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=None,
        metavar="CLASS",
        help="Canonical class names to download (default: all classes in class_mapping.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        metavar="N",
        help="Max images per class (default: 200)",
    )

    # Roboflow-specific
    parser.add_argument("--url",     default=None, help="[roboflow] Dataset URL")
    parser.add_argument("--version", type=int, default=1, help="[roboflow] Dataset version")
    parser.add_argument("--api-key", default=None, help="[roboflow] API key (or set ROBOFLOW_API_KEY env var)")

    # Kaggle-specific
    parser.add_argument("--dataset", default=None, help="[kaggle] Dataset slug: user/dataset-name")

    # Manual-specific
    parser.add_argument("--zip",  default=None, help="[manual] Path to local ZIP file")
    parser.add_argument("--name", default=None, help="[manual] Name for the dataset subfolder")

    return parser


def main() -> None:
    providers = load_providers()
    enabled   = get_enabled_providers(providers)

    if not enabled:
        print("No enabled providers found in dataset_sources.yaml.")
        sys.exit(1)

    parser = build_parser(enabled)
    args   = parser.parse_args()

    provider_cfg = enabled[args.provider]
    module_path  = f"scripts.{provider_cfg['module']}"

    # Import the provider module dynamically
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"ERROR: Could not import provider module '{module_path}': {e}")
        print("Make sure the provider file exists in scripts/providers/")
        sys.exit(1)

    if not hasattr(module, "download"):
        print(f"ERROR: {module_path} does not implement a download(args) function.")
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print(f"AgriVision Dataset Downloader")
    print(f"Provider : {args.provider}")
    print(f"{'=' * 50}\n")

    module.download(args)


if __name__ == "__main__":
    main()
