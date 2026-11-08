from __future__ import annotations

import csv
from pathlib import Path

from common.io import list_dataset_images
from common.pipeline_runner import RESULTS_DIR, ROOT_DIR, load_config, run_pipeline

DATA_DIR = ROOT_DIR / "data" / "samples"
SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
FIELDNAMES = ["member", "pipeline", "image", "mse_vs_original", "psnr_vs_original", "ssim_vs_original"]


def run_all() -> list[dict]:
    """Запускає кожен пайплайн кожного учасника на кожному зображенні вибірки —
    спільне порівняння всіх пайплайнів команди (блок C)."""
    config = load_config()
    images = list_dataset_images(DATA_DIR)
    if not images:
        raise SystemExit(
            f"Немає зображень у {DATA_DIR}. Додайте вибірку (див. data/README.md) "
            "перед запуском повного порівняння."
        )

    all_metrics: list[dict] = []
    for member, pipelines in config["members"].items():
        for pipeline_name in pipelines:
            for image_path in images:
                metrics = run_pipeline(member, pipeline_name, image_path)
                all_metrics.append(metrics)
                print(
                    f"{member}/{pipeline_name} на {image_path.name}: "
                    f"MSE={metrics['mse_vs_original']} "
                    f"PSNR={metrics['psnr_vs_original']} "
                    f"SSIM={metrics['ssim_vs_original']}"
                )
    return all_metrics


def save_summary(all_metrics: list[dict]) -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_metrics)
    return SUMMARY_PATH


def main() -> None:
    all_metrics = run_all()
    summary_path = save_summary(all_metrics)
    print(f"\nЗбережено зведені метрики ({len(all_metrics)} рядків): {summary_path}")


if __name__ == "__main__":
    main()
