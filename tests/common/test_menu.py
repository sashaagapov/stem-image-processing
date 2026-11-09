from PIL import Image

from common import menu


def test_list_pipelines_numbers_all_pipelines_in_config_order():
    config = {"members": {
        "member1": {"pipeline1": {"description": "A -> B -> C"}, "pipeline2": {"description": "D -> E -> F"}},
        "member2": {"pipeline1": {"description": "G -> H -> I"}},
    }}
    pipelines = menu.list_pipelines(config)
    assert [(p["number"], p["member"], p["key"]) for p in pipelines] == [
        (1, "member1", "pipeline1"), (2, "member1", "pipeline2"), (3, "member2", "pipeline1"),
    ]
    assert pipelines[2]["stages"] == ["1. G", "2. H", "3. I"]
    assert pipelines[2]["author"] == "Тимофій Стукало"


def test_real_config_has_six_pipelines():
    assert len(menu.list_pipelines(menu.load_config())) == 6


def test_resolve_image_path_accepts_quoted_full_path(tmp_path):
    image = tmp_path / "scene.png"
    Image.new("RGB", (4, 4)).save(image)
    assert menu.resolve_image_path(f'"{image}"') == image


def test_resolve_image_path_finds_sample_by_name(tmp_path, monkeypatch):
    Image.new("RGB", (4, 4)).save(tmp_path / "river_02.jpg")
    monkeypatch.setattr(menu, "SAMPLES_DIR", tmp_path)
    assert menu.resolve_image_path("river_02") == tmp_path / "river_02.jpg"
    assert menu.resolve_image_path("", "river_02.jpg") == tmp_path / "river_02.jpg"


def test_resolve_image_path_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(menu, "SAMPLES_DIR", tmp_path)
    assert menu.resolve_image_path("nope.jpg", "nope.jpg") is None


def test_build_comparison_puts_all_images_in_one_row():
    original = Image.new("RGB", (8, 8), "red")
    stages = [Image.new("L", (8, 8), 100) for _ in range(3)]
    sheet = menu.build_comparison(original, stages, ["a", "b", "c"], size=20)
    # 4 картинки по 20 px + 5 проміжків по 10 px
    assert sheet.size == (4 * 20 + 5 * 10, 20 + 30 + 20)
    assert sheet.getpixel((10, 10)) == (255, 0, 0)
