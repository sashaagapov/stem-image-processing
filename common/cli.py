from __future__ import annotations

import argparse
from pathlib import Path

from common.pipeline_runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустити один пайплайн на одному зображенні.")
    parser.add_argument("--member", required=True, help="напр. member1")
    parser.add_argument("--pipeline", required=True, help="напр. pipeline1")
    parser.add_argument("--image", required=True, type=Path, help="шлях до вхідного зображення")
    args = parser.parse_args()

    metrics = run_pipeline(args.member, args.pipeline, args.image)

    print(f"Готово: {args.member}/{args.pipeline} на {args.image.name}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
