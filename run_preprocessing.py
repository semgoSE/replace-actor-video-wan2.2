#!/usr/bin/env python3
"""
Этап 1: Preprocessing.

Запускает официальный препроцессинг Wan2.2 (preprocess_data.py):
- вход: видео + референсное изображение
- выход: папка (save_path) с подготовленными данными для генерации

После preprocess в save_path появляются:
  Animate: src_ref.png, src_face.mp4, src_pose.mp4
  Replace: + src_bg.mp4, src_mask.mp4

Требуется: клонированный репозиторий Wan2.2 и скачанный Wan2.2-Animate-14B
(в нём должна быть подпапка process_checkpoint).

Использование:
  python run_preprocessing.py --video_path dance.mp4 --refer_path face.jpg --save_path process_results
  python run_preprocessing.py --config config.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_wan22_preprocess(wan22_dir: Path) -> Path:
    script = wan22_dir / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Скрипт препроцессинга не найден: {script}\n"
            "Клонируйте репозиторий: git clone https://github.com/Wan-Video/Wan2.2.git"
        )
    return script


def run_preprocessing(
    wan22_dir: Path,
    ckpt_dir: Path,
    video_path: Path,
    refer_path: Path,
    save_path: Path,
    resolution_area: tuple = (1280, 720),
    retarget_flag: bool = True,
    replace_flag: bool = False,
    use_flux: bool = False,
    iterations: int = 3,
    k: int = 7,
    w_len: int = 1,
    h_len: int = 1,
) -> int:
    """
    Запускает preprocess_data.py из Wan2.2.
    Возвращает код возврата процесса (0 = успех).
    """
    script = find_wan22_preprocess(wan22_dir)
    process_ckpt = ckpt_dir / "process_checkpoint"
    if not process_ckpt.exists():
        raise FileNotFoundError(
            f"Не найдена папка process_checkpoint: {process_ckpt}\n"
            "Скачайте модель: huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B"
        )
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if not refer_path.exists():
        raise FileNotFoundError(f"Референс не найден: {refer_path}")

    save_path.mkdir(parents=True, exist_ok=True)
    w, h = resolution_area[0], resolution_area[1]

    cmd = [
        sys.executable,
        str(script),
        "--ckpt_path", str(process_ckpt),
        "--video_path", str(video_path.resolve()),
        "--refer_path", str(refer_path.resolve()),
        "--save_path", str(save_path.resolve()),
        "--resolution_area", str(w), str(h),
    ]
    if retarget_flag:
        cmd.append("--retarget_flag")
    if replace_flag:
        cmd.append("--replace_flag")
        cmd.extend(["--iterations", str(iterations), "--k", str(k), "--w_len", str(w_len), "--h_len", str(h_len)])
    if use_flux:
        cmd.append("--use_flux")

    print("Этап 1: Preprocessing (Wan2.2 preprocess_data.py)")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(wan22_dir))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Этап 1: Preprocessing для Wan2.2-Animate")
    parser.add_argument("--wan22_dir", type=Path, default=Path("Wan2.2"), help="Путь к репозиторию Wan2.2")
    parser.add_argument("--ckpt_dir", type=Path, default=Path("Wan2.2-Animate-14B"), help="Путь к чекпоинту (с process_checkpoint)")
    parser.add_argument("--video_path", type=Path, required=True, help="Входное видео (поза/движение)")
    parser.add_argument("--refer_path", type=Path, required=True, help="Референсное изображение персонажа")
    parser.add_argument("--save_path", type=Path, default=Path("process_results"), help="Папка для результата препроцессинга")
    parser.add_argument("--resolution", type=str, default="1280 720", help="Разрешение: W H (например 1280 720)")
    parser.add_argument("--retarget", action="store_true", default=True, help="Режим animate (персонаж повторяет движение)")
    parser.add_argument("--replace", action="store_true", help="Режим replace (замена актёра в видео)")
    parser.add_argument("--use_flux", action="store_true", help="Использовать FLUX в препроцессинге")
    parser.add_argument("--iterations", type=int, default=3, help="Для replace: iterations")
    parser.add_argument("--k", type=int, default=7, help="Для replace: k")
    args = parser.parse_args()

    res = [int(x) for x in args.resolution.split()]
    resolution_area = (res[0], res[1]) if len(res) >= 2 else (1280, 720)

    ret = run_preprocessing(
        wan22_dir=args.wan22_dir,
        ckpt_dir=args.ckpt_dir,
        video_path=args.video_path,
        refer_path=args.refer_path,
        save_path=args.save_path,
        resolution_area=resolution_area,
        retarget_flag=args.retarget and not args.replace,
        replace_flag=args.replace,
        use_flux=args.use_flux,
        iterations=args.iterations,
        k=args.k,
    )
    if ret == 0:
        print("Preprocessing завершён. Результат в:", args.save_path.resolve())
    sys.exit(ret)


if __name__ == "__main__":
    main()
