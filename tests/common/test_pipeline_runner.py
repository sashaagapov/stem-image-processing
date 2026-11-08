import shutil
from pathlib import Path

from PIL import Image

from common.io import open_image, save_image
from common.pipeline_runner import RESULTS_DIR, run_pipeline


def test_run_pipeline_executes_all_stages_and_saves_outputs(tmp_path: Path) -> None:
    """Smoke test: з заглушками apply() (поки реальні методи не реалізовані)
    весь пайплайн має відпрацювати без помилок і зберегти результати."""
    source_path = tmp_path / "source.png"
    save_image(Image.new("RGB", (16, 16), (100, 150, 200)), source_path)

    output_dir = RESULTS_DIR / "member1" / "pipeline1" / source_path.stem
    shutil.rmtree(output_dir, ignore_errors=True)

    try:
        metrics = run_pipeline("member1", "pipeline1", source_path)

        assert metrics["member"] == "member1"
        assert metrics["pipeline"] == "pipeline1"
        assert (output_dir / "stage_1_stage_1_sobel.png").exists()
        assert (output_dir / "stage_2_stage_2_histogram_equalization.png").exists()
        assert (output_dir / "stage_3_stage_3_geometric_transform.png").exists()
        assert (output_dir / "final.png").exists()
        assert (output_dir / "metrics.json").exists()

        final_image = open_image(output_dir / "final.png")
        assert final_image.size == (16, 16)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_run_pipeline_rejects_unknown_pipeline(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    save_image(Image.new("RGB", (8, 8)), source_path)

    try:
        run_pipeline("member1", "pipeline_that_does_not_exist", source_path)
    except ValueError:
        return
    assert False, "Очікувався ValueError для невідомого пайплайну"
