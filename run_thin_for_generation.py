#!/usr/bin/env python3
"""
Прореживание кадров препроцессинга для генерации (30 fps → 15 fps, каждый 2-й кадр).

Используется в связке с workflow:
  - Preprocess на 30 fps (DWPose видит все кадры, полная поза).
  - Thin: берём каждый 2-й кадр → 15 fps (экономия времени генерации).
  - Generation по прореженной папке.
  - RIFE восстанавливает 15 → 30 fps.

Вход: папка с результатом препроцессинга (src_pose.mp4, src_face.mp4, src_ref.png; для replace + src_bg.mp4, src_mask.mp4).
Выход: новая папка с теми же файлами, но видео с каждым 2-м кадром и 15 fps.

Использование:
  python run_thin_for_generation.py --input preprocess/my_run --output preprocess/my_run_gen
  python run_thin_for_generation.py --input preprocess/my_run  # по умолчанию output = <input>_gen
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Видео в папке препроцессинга, которые нужно проредить (каждый 2-й кадр)
THIN_VIDEOS = ("src_pose.mp4", "src_face.mp4", "src_bg.mp4", "src_mask.mp4")
COPY_FILES = ("src_ref.png",)


def _thin_video_every_nth(src: Path, dst: Path, nth: int = 2, out_fps: float = 15) -> bool:
    """Оставляет каждый n-й кадр, перекодирует в out_fps. Возвращает True при успехе."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # select: not(mod(n\,nth)) — кадры 0, nth, 2*nth, ...
    # -vsync cfr -r out_fps — постоянный выходной fps
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vf", f"select='not(mod(n\\,{nth}))'",
        "-vsync", "cfr",
        "-r", str(out_fps),
        "-an",
        str(dst),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.returncode == 0 and dst.exists()
    except Exception:
        return False


def run_thin(
    input_dir: Path,
    output_dir: Path,
    every_nth: int = 2,
    out_fps: float = 15,
) -> int:
    """
    Создаёт в output_dir копию препроцессинга с прореженными видео (каждый every_nth кадр, out_fps).
    Возвращает 0 при успехе.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.is_dir():
        print(f"Ошибка: папка не найдена: {input_dir}", file=sys.stderr)
        return 1

    src_pose = input_dir / "src_pose.mp4"
    if not src_pose.exists():
        print(f"Ошибка: в папке нет src_pose.mp4 (это не папка препроцессинга?): {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in COPY_FILES:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    for name in THIN_VIDEOS:
        src = input_dir / name
        if not src.exists():
            continue
        dst = output_dir / name
        print(f"Прореживание: {name} (каждый {every_nth}-й кадр → {out_fps} fps)")
        if not _thin_video_every_nth(src, dst, nth=every_nth, out_fps=out_fps):
            print(f"Ошибка при прореживании {name}", file=sys.stderr)
            return 1

    print("Готово. Папка для генерации:", output_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Прореживание кадров препроцессинга (30→15 fps) для ускорения генерации"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Папка с результатом препроцессинга (30 fps)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Папка для прореженных данных (по умолчанию <input>_gen)",
    )
    parser.add_argument(
        "--every_nth",
        type=int,
        default=2,
        help="Брать каждый n-й кадр (по умолч. 2)",
    )
    parser.add_argument(
        "--out_fps",
        type=float,
        default=15,
        help="FPS выходных видео (по умолч. 15)",
    )
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = Path(str(args.input).rstrip("/"))  # preprocess/my_run
        out = out.parent / (out.name + "_gen")   # preprocess/my_run_gen

    return run_thin(args.input, out, every_nth=args.every_nth, out_fps=args.out_fps)


if __name__ == "__main__":
    sys.exit(main())
