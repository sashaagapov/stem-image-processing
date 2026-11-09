import shutil

import numpy as np
import pytest
from PIL import Image

from common import gui
from common.pipeline_runner import RESULTS_DIR


def test_pipeline_label_shows_number_stages_and_author():
    pipeline = {"number": 3, "author": "Тимофій Стукало", "stages": ["1. A", "2. B", "3. C"]}
    assert gui.pipeline_label(pipeline) == "3. A → B → C  (Тимофій Стукало)"


def test_process_returns_original_and_three_stage_images(tmp_path):
    rng = np.random.default_rng(0)
    image_path = tmp_path / "gui_test_scene.png"
    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(image_path)
    pipeline = next(p for p in gui.menu.list_pipelines(gui.load_config()) if p["number"] == 3)
    try:
        result = gui.process(pipeline, image_path)
        assert len(result["images"]) == 4
        assert (result["output_dir"] / "comparison.png").is_file()
        assert set(result["metrics"]) >= {"mse_vs_original", "psnr_vs_original", "ssim_vs_original"}
    finally:
        shutil.rmtree(RESULTS_DIR / "member2" / "pipeline1" / "gui_test_scene", ignore_errors=True)


def test_window_builds_with_all_six_pipelines():
    try:
        root = gui.tk.Tk()
    except gui.tk.TclError:
        pytest.skip("немає графічного середовища")
    try:
        app = gui.App(root)
        assert len(app.pipeline_box["values"]) == 6
        assert app.pipeline_box.current() == 2
        root.update_idletasks()
        # порожні рамки під картинки мають бути в пікселях, а не в символах (вікно не на весь екран)
        assert root.winfo_reqwidth() < 1400 and root.winfo_reqheight() < 800
    finally:
        root.destroy()
