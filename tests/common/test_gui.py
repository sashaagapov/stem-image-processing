import shutil

import numpy as np
import pytest
from PIL import Image

from common import gui
from common.pipeline_runner import RESULTS_DIR


def test_list_pipelines_numbers_all_pipelines_in_config_order():
    config = {"members": {
        "member1": {"pipeline1": {"description": "A -> B -> C"}, "pipeline2": {"description": "D -> E -> F"}},
        "member2": {"pipeline1": {"description": "G -> H -> I"}},
    }}
    pipelines = gui.list_pipelines(config)
    assert [(p["number"], p["member"], p["key"]) for p in pipelines] == [
        (1, "member1", "pipeline1"), (2, "member1", "pipeline2"), (3, "member2", "pipeline1"),
    ]
    assert pipelines[2]["stages"] == ["1. G", "2. H", "3. I"]
    assert pipelines[2]["author"] == "Тимофій Стукало"


def test_real_config_has_six_pipelines():
    assert len(gui.list_pipelines(gui.load_config())) == 6


def test_pipeline_label_shows_number_stages_and_author():
    pipeline = {"number": 3, "author": "Тимофій Стукало", "stages": ["1. A", "2. B", "3. C"]}
    assert gui.pipeline_label(pipeline) == "3. A → B → C  (Тимофій Стукало)"


def test_list_sample_images_returns_sorted_jpg_names(tmp_path, monkeypatch):
    for name in ["river_02.jpg", "agricultural_01.jpg", "notes.txt"]:
        Image.new("RGB", (4, 4)).save(tmp_path / name) if name.endswith(".jpg") else (tmp_path / name).write_text("x")
    monkeypatch.setattr(gui, "SAMPLES_DIR", tmp_path)
    assert gui.list_sample_images() == ["agricultural_01.jpg", "river_02.jpg"]


def test_list_sample_images_handles_missing_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "SAMPLES_DIR", tmp_path / "does_not_exist")
    assert gui.list_sample_images() == []


def test_resolve_image_path_accepts_quoted_full_path(tmp_path):
    image = tmp_path / "scene.png"
    Image.new("RGB", (4, 4)).save(image)
    assert gui.resolve_image_path(f'"{image}"') == image


def test_resolve_image_path_finds_sample_by_name(tmp_path, monkeypatch):
    Image.new("RGB", (4, 4)).save(tmp_path / "river_02.jpg")
    monkeypatch.setattr(gui, "SAMPLES_DIR", tmp_path)
    assert gui.resolve_image_path("river_02") == tmp_path / "river_02.jpg"
    assert gui.resolve_image_path("", "river_02.jpg") == tmp_path / "river_02.jpg"


def test_resolve_image_path_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "SAMPLES_DIR", tmp_path)
    assert gui.resolve_image_path("nope.jpg", "nope.jpg") is None


def test_process_returns_original_and_three_stage_images(tmp_path):
    rng = np.random.default_rng(0)
    image_path = tmp_path / "gui_test_scene.png"
    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(image_path)
    pipeline = next(p for p in gui.list_pipelines(gui.load_config()) if p["number"] == 3)
    try:
        result = gui.process(pipeline, image_path)
        assert len(result["images"]) == 4
        assert (result["output_dir"] / "final.png").is_file()
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
        # блок статистики пайплайну 2 не займає місця, поки пайплайн 2 не запущено
        assert not app.stats_frame.grid_info()
    finally:
        root.destroy()


def test_show_otsu_statistics_matches_the_real_pipeline_functions():
    try:
        root = gui.tk.Tk()
    except gui.tk.TclError:
        pytest.skip("немає графічного середовища")
    try:
        app = gui.App(root)
        rng = np.random.default_rng(1)
        original = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        color_corrected = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        mask_array = (rng.integers(0, 2, (16, 16)) * 255).astype(np.uint8)
        mask = Image.fromarray(np.stack([mask_array] * 3, axis=-1))
        statistics_thumb = mask.copy()  # заглушка замість реального stage_3.png з дрібним текстом

        pipeline = next(p for p in app.pipelines if p["number"] == 2)
        app._show_result(pipeline, gui.SAMPLES_DIR / "river_02.jpg", {
            "images": [original, color_corrected, mask, statistics_thumb],
            "metrics": {"mse_vs_original": 0, "psnr_vs_original": 0, "ssim_vs_original": 0},
            "output_dir": gui.SAMPLES_DIR,
        })

        # ті самі функції, якими рахує сама стадія — число має збігатися до символу.
        array = np.asarray(color_corrected.convert("RGB"), dtype=np.float64)
        expected_threshold = gui.find_otsu_threshold(gui.otsu_grayscale(array))
        expected_stats = gui.calculate_statistics(mask)

        assert app.stats_frame.grid_info()  # блок показано
        text = app.stats_label["text"]
        assert str(expected_threshold) in text
        assert f"{expected_stats['mean']:.2f}" in text
        assert f"{expected_stats['variance']:.2f}" in text
        assert f"{expected_stats['entropy']:.3f}" in text
    finally:
        root.destroy()


def test_other_pipelines_keep_stats_frame_hidden():
    try:
        root = gui.tk.Tk()
    except gui.tk.TclError:
        pytest.skip("немає графічного середовища")
    try:
        app = gui.App(root)
        app.stats_frame.grid()  # імітує, що попереднім пайплайном міг бути пайплайн 2
        pipeline = next(p for p in app.pipelines if p["number"] == 1)
        app._show_result(pipeline, gui.SAMPLES_DIR / "river_02.jpg",
                          {"images": [Image.new("RGB", (4, 4))] * 4,
                           "metrics": {"mse_vs_original": 0, "psnr_vs_original": 0, "ssim_vs_original": 0},
                           "output_dir": gui.SAMPLES_DIR})
        assert not app.stats_frame.grid_info()
    finally:
        root.destroy()
