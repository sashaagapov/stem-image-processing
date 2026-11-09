"""Віконна програма (Tkinter) для запуску всіх 6 пайплайнів команди.

Запуск: кнопка «Пуск» у Visual Studio (Startup Item — common/gui.py) або
    .venv\\Scripts\\python.exe common\\gui.py
Логіка (список пайплайнів, пошук знімка) спільна з консольним меню common/menu.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

# Якщо вікно запустили не тим Python, перезапускаємо його через .venv проєкту
# (до імпорту numpy/yaml, яких в іншому Python може не бути).
if __name__ == "__main__" and VENV_PYTHON.exists() \
        and Path(sys.prefix).resolve() != VENV_PYTHON.parents[1].resolve():
    raise SystemExit(subprocess.call([str(VENV_PYTHON), str(Path(__file__).resolve())]))

sys.path.insert(0, str(ROOT_DIR))  # щоб працювали імпорти common.* і src.*

import tkinter as tk  # noqa: E402

from common import menu  # noqa: E402
from tkinter import filedialog, ttk  # noqa: E402

from PIL import Image, ImageTk  # noqa: E402

from common.io import ImageLoadError, open_image  # noqa: E402
from common.pipeline_runner import RESULTS_DIR, load_config, run_pipeline  # noqa: E402

THUMB_SIZE = 230


def pipeline_label(pipeline: dict) -> str:
    """Підпис пайплайна у списку: номер, стадії через стрілки, автор."""
    stages = " → ".join(stage.split(". ", 1)[1] for stage in pipeline["stages"])
    return f"{pipeline['number']}. {stages}  ({pipeline['author']})"


def process(pipeline: dict, image_path: Path) -> dict:
    """Запускає пайплайн і повертає метрики, папку результатів і картинки (оригінал + стадії)."""
    metrics = run_pipeline(pipeline["member"], pipeline["key"], image_path)
    output_dir = RESULTS_DIR / pipeline["member"] / pipeline["key"] / image_path.stem
    images = [open_image(image_path)] + [open_image(p) for p in sorted(output_dir.glob("stage_*.png"))]
    menu.build_comparison(images[0], images[1:], pipeline["stages"]).save(output_dir / "comparison.png")
    return {"metrics": metrics, "output_dir": output_dir, "images": images}


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.pipelines = menu.list_pipelines(load_config())
        self.output_dir: Path | None = None
        self._photos: list[ImageTk.PhotoImage] = []  # Tkinter не тримає посилання на картинки сам

        root.title("Аналіз супутникових знімків — STEM-проєкт")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=12)
        frame.grid()
        frame.columnconfigure(1, weight=1)  # зайва ширина йде полям вводу, а не підписам

        ttk.Label(frame, text="Пайплайн:").grid(row=0, column=0, sticky="w")
        self.pipeline_box = ttk.Combobox(frame, state="readonly", width=95,
                                         values=[pipeline_label(p) for p in self.pipelines])
        self.pipeline_box.current(2)  # пайплайн 3 — перший пайплайн member2
        self.pipeline_box.grid(row=0, column=1, columnspan=3, sticky="we", pady=2)

        ttk.Label(frame, text="Знімок:").grid(row=1, column=0, sticky="w")
        self.image_var = tk.StringVar(value=str(menu.SAMPLES_DIR / menu.DEFAULT_IMAGE))
        ttk.Entry(frame, textvariable=self.image_var, width=80).grid(row=1, column=1, columnspan=2, sticky="we", pady=2)
        ttk.Button(frame, text="Обрати файл…", command=self.choose_file).grid(row=1, column=3, sticky="e", padx=(6, 0))

        self.run_button = ttk.Button(frame, text="▶ Запустити", command=self.start)
        self.run_button.grid(row=2, column=3, sticky="e", pady=8)
        self.status = ttk.Label(frame, text="Оберіть пайплайн і знімок, потім натисніть «Запустити».")
        self.status.grid(row=2, column=0, columnspan=3, sticky="w")

        gallery = ttk.Frame(frame)
        gallery.grid(row=3, column=0, columnspan=4, pady=6)
        self.image_labels, self.caption_labels = [], []
        for column in range(4):
            image_label = tk.Label(gallery, width=THUMB_SIZE, height=THUMB_SIZE, bg="#e6e6e6")
            image_label.grid(row=0, column=column, padx=4)
            caption = ttk.Label(gallery, text="", wraplength=THUMB_SIZE, justify="center")
            caption.grid(row=1, column=column, pady=(2, 0))
            self.image_labels.append(image_label)
            self.caption_labels.append(caption)

        self.metrics_label = ttk.Label(frame, text="", font=("Segoe UI", 11, "bold"))
        self.metrics_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.folder_label = ttk.Label(frame, text="")
        self.folder_label.grid(row=5, column=0, columnspan=3, sticky="w")
        self.folder_button = ttk.Button(frame, text="Відкрити папку", command=self.open_folder, state="disabled")
        self.folder_button.grid(row=5, column=3, sticky="e")

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Оберіть знімок", initialdir=menu.SAMPLES_DIR,
            filetypes=[("Зображення", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Усі файли", "*.*")])
        if path:
            self.image_var.set(path)

    def start(self) -> None:
        pipeline = self.pipelines[self.pipeline_box.current()]
        image_path = menu.resolve_image_path(self.image_var.get())
        if image_path is None:
            self.status.config(text="Файл не знайдено. Перевірте шлях або оберіть файл кнопкою.", foreground="red")
            return
        self.run_button.config(state="disabled")
        self.status.config(text=f"Обробка {image_path.name}…", foreground="black")
        # Пайплайн працює в окремому потоці, щоб вікно не зависало.
        threading.Thread(target=self._work, args=(pipeline, image_path), daemon=True).start()

    def _work(self, pipeline: dict, image_path: Path) -> None:
        try:
            result = process(pipeline, image_path)
        except (ImageLoadError, ValueError) as error:
            self.root.after(0, self._show_error, str(error))
            return
        self.root.after(0, self._show_result, pipeline, image_path, result)

    def _show_error(self, message: str) -> None:
        self.run_button.config(state="normal")
        self.status.config(text=f"Помилка: {message}", foreground="red")

    def _show_result(self, pipeline: dict, image_path: Path, result: dict) -> None:
        self.run_button.config(state="normal")
        self.status.config(text=f"Готово: пайплайн {pipeline['number']} на {image_path.name}", foreground="green")
        captions = ["Оригінал"] + pipeline["stages"]
        self._photos = []
        for index, label in enumerate(self.image_labels):
            if index < len(result["images"]):
                thumb = result["images"][index].convert("RGB").resize((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)
                photo = ImageTk.PhotoImage(thumb)
                self._photos.append(photo)
                label.config(image=photo, width=THUMB_SIZE, height=THUMB_SIZE)
                self.caption_labels[index].config(text=captions[index])
            else:
                label.config(image="")
                self.caption_labels[index].config(text="")
        metrics = result["metrics"]
        self.metrics_label.config(text=f"MSE = {metrics['mse_vs_original']}     "
                                       f"PSNR = {metrics['psnr_vs_original']} дБ     "
                                       f"SSIM = {metrics['ssim_vs_original']}")
        self.output_dir = result["output_dir"]
        self.folder_label.config(text=f"Результати: {self.output_dir.relative_to(ROOT_DIR)}")
        self.folder_button.config(state="normal")

    def open_folder(self) -> None:
        if self.output_dir:
            os.startfile(self.output_dir)


def main() -> None:
    os.chdir(ROOT_DIR)
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
