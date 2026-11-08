from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from common.io import open_image, save_image
from common.metrics import calculate_mse, calculate_psnr, calculate_ssim

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "pipelines.yaml"
RESULTS_DIR = ROOT_DIR / "results"


@dataclass
class StageSpec:
    module: str
    function: str
    params: dict


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_pipeline_stages(member: str, pipeline: str) -> list[StageSpec]:
    config = load_config()
    try:
        pipeline_config = config["members"][member][pipeline]
    except KeyError as error:
        raise ValueError(f"Пайплайн {member}/{pipeline} не знайдено у config/pipelines.yaml") from error

    return [
        StageSpec(
            module=stage["module"],
            function=stage.get("function", "apply"),
            params=stage.get("params", {}) or {},
        )
        for stage in pipeline_config["stages"]
    ]


def run_pipeline(member: str, pipeline: str, image_path: Path) -> dict:
    """Виконує всі стадії пайплайну по черзі (стадія N отримує результат стадії N-1),
    зберігає проміжні й фінальний результат та рахує метрики відносно вхідного зображення."""
    stages = get_pipeline_stages(member, pipeline)
    original = open_image(image_path)
    current = original

    output_dir = RESULTS_DIR / member / pipeline / image_path.stem

    for index, stage in enumerate(stages, start=1):
        module = importlib.import_module(stage.module)
        function = getattr(module, stage.function)
        current = function(current, **stage.params)

        stage_name = stage.module.rsplit(".", 1)[-1]
        save_image(current, output_dir / f"stage_{index}_{stage_name}.png")

    save_image(current, output_dir / "final.png")

    metrics = {
        "member": member,
        "pipeline": pipeline,
        "image": image_path.name,
        "mse_vs_original": round(calculate_mse(current, original), 4),
        "psnr_vs_original": round(calculate_psnr(current, original), 4),
        "ssim_vs_original": round(calculate_ssim(current, original), 4),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics
