"""
Конфиг пайплайна Wan2.2-Animate: два этапа (preprocessing → generation).
Чисто Python, без ComfyUI. Использует официальный Wan2.2 репозиторий.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    # === Путь к репозиторию Wan2.2 (клонированный) ===
    wan22_dir: Path = Path("Wan2.2")

    # === Путь к чекпоинту Wan2.2-Animate-14B ===
    ckpt_dir: Path = Path("Wan2.2-Animate-14B")

    # === Этап 1: Preprocessing ===
    video_path: Path = Path("video.mp4")           # видео с позой/движением
    refer_path: Path = Path("reference.jpg")       # референсное фото персонажа
    save_path: Path = Path("process_results")      # папка результата препроцессинга
    resolution_area: tuple = (1280, 720)           # (width, height) или площадь
    retarget_flag: bool = True                      # True = animate (персонаж повторяет движение)
    replace_flag: bool = False                      # True = replace (замена актёра в видео)
    use_flux: bool = False                         # опция препроцессинга
    # для replace_mode:
    iterations: int = 3
    k: int = 7
    w_len: int = 1
    h_len: int = 1

    # === Этап 2: Generation ===
    refert_num: int = 1
    use_relighting_lora: bool = False
    output_path: Path = Path("output_wanimate.mp4")
    seed: int = 42
    # Multi-GPU (опционально)
    multi_gpu: bool = False
    nproc_per_node: int = 8

    def __post_init__(self):
        for name in ("wan22_dir", "ckpt_dir", "video_path", "refer_path", "save_path", "output_path"):
            v = getattr(self, name)
            if not isinstance(v, Path):
                setattr(self, name, Path(v))
