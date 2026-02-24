#!/usr/bin/env python3
"""
Этап 4 (опционально): Upscale — SeedVR2 или Real-ESRGAN.

Принимает видео после постобработки (например 30 fps от run_postprocess.py).

  seedvr2: одношаговое восстановление до 720p/1080p/2K (тяжёлая модель, лучше качество).
  realesrgan: апскейл 2×/4× по кадрам (легче, быстрее).

Использование:
  python run_upscale.py --input video_30fps.mp4 --output video_720p.mp4
  python run_upscale.py --input video_30fps.mp4 --output video_720p.mp4 --backend realesrgan --outscale 2
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_input_fps(path: Path) -> float | None:
    """FPS входного видео через ffprobe."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rate = out.stdout.strip().strip('"')
        if "/" in rate:
            num, den = rate.split("/")
            return float(num) / float(den) if float(den) != 0 else None
        return float(rate)
    except Exception:
        return None


def run_upscale(
    input_video: Path,
    output_video: Path,
    seedvr_dir: Path,
    res_h: int = 720,
    res_w: int = 1280,
    num_gpus: int = 1,
    sp_size: int | None = None,
    seed: int = 42,
    keep_audio: bool = True,
) -> int:
    """
    Апскейл через SeedVR2: входное видео → папка → torchrun inference → сборка с аудио.
    """
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    seedvr_dir = Path(seedvr_dir).resolve()

    if not input_video.exists():
        print(f"Ошибка: файл не найден: {input_video}", file=sys.stderr)
        return 1
    infer_script = seedvr_dir / "projects" / "inference_seedvr2_3b.py"
    if not infer_script.exists():
        print(
            f"Ошибка: скрипт не найден: {infer_script}\n"
            "Клонируйте: git clone https://github.com/ByteDance-Seed/SeedVR",
            file=sys.stderr,
        )
        return 1

    # SeedVR принимает папку с видео; выход — в другую папку с тем же именем файла
    with tempfile.TemporaryDirectory(prefix="seedvr_in_") as tmp_in:
        with tempfile.TemporaryDirectory(prefix="seedvr_out_") as tmp_out:
            # один файл в папке (скрипт делает listdir и читает каждый файл как видео)
            link_path = Path(tmp_in) / input_video.name
            try:
                link_path.symlink_to(input_video)
            except OSError:
                shutil.copy2(input_video, link_path)

            nproc = max(1, num_gpus)
            sp = sp_size if sp_size is not None else nproc
            out_fps = get_input_fps(input_video)

            cmd = [
                "torchrun",
                f"--nproc-per-node={nproc}",
                str(infer_script),
                "--video_path", tmp_in,
                "--output_dir", tmp_out,
                "--res_h", str(res_h),
                "--res_w", str(res_w),
                "--sp_size", str(sp),
                "--seed", str(seed),
            ]
            if out_fps is not None:
                cmd += ["--out_fps", str(out_fps)]

            print("SeedVR2 upscale...", cmd)
            r = subprocess.run(cmd, cwd=str(seedvr_dir))
            if r.returncode != 0:
                return r.returncode

            upscaled = Path(tmp_out) / input_video.name
            if not upscaled.exists():
                print(f"Ошибка: SeedVR не создал файл {upscaled}", file=sys.stderr)
                return 1

            output_video.parent.mkdir(parents=True, exist_ok=True)
            if keep_audio:
                # подмешиваем аудио из исходного видео (SeedVR пишет только видео)
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(upscaled),
                    "-i", str(input_video),
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    str(output_video),
                ], check=True)
            else:
                shutil.copy2(upscaled, output_video)

    print("Готово:", output_video)
    return 0


def run_realesrgan_upscale(
    input_video: Path,
    output_video: Path,
    realesrgan_dir: Path,
    model_name: str = "realesr-general-x4v3",
    outscale: float = 2,
    tile: int = 0,
    face_enhance: bool = False,
    fp32: bool = False,
) -> int:
    """
    Апскейл через Real-ESRGAN: вызов inference_realesrgan_video.py (видео → папка вывода → копируем файл).
    """
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    realesrgan_dir = Path(realesrgan_dir).resolve()

    if not input_video.exists():
        print(f"Ошибка: файл не найден: {input_video}", file=sys.stderr)
        return 1
    infer_script = realesrgan_dir / "inference_realesrgan_video.py"
    if not infer_script.exists():
        print(
            f"Ошибка: скрипт не найден: {infer_script}\n"
            "Клонируйте: git clone https://github.com/xinntao/Real-ESRGAN",
            file=sys.stderr,
        )
        return 1

    output_video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="realesrgan_out_") as tmp_out:
        cmd = [
            sys.executable,
            str(infer_script),
            "-i", str(input_video),
            "-o", tmp_out,
            "-n", model_name,
            "-s", str(outscale),
            "-t", str(tile),
        ]
        if face_enhance:
            cmd.append("--face_enhance")
        if fp32:
            cmd.append("--fp32")

        print("Real-ESRGAN upscale...", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(realesrgan_dir))
        if r.returncode != 0:
            return r.returncode

        # скрипт сохраняет как {video_name}_out.mp4 в папку вывода
        base_name = input_video.stem
        upscaled = Path(tmp_out) / f"{base_name}_out.mp4"
        if not upscaled.exists():
            print(f"Ошибка: Real-ESRGAN не создал файл {upscaled}", file=sys.stderr)
            return 1
        shutil.copy2(upscaled, output_video)

    print("Готово:", output_video)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Upscale видео (SeedVR2 или Real-ESRGAN) после run_postprocess"
    )
    parser.add_argument("--input", type=Path, required=True, help="Входное видео (например после RIFE 30 fps)")
    parser.add_argument("--output", type=Path, required=True, help="Выходное видео (upscale)")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["seedvr2", "realesrgan"],
        default="seedvr2",
        help="seedvr2 (720p/1080p/2K) или realesrgan (2×/4× по кадрам)",
    )
    # SeedVR2
    parser.add_argument("--seedvr_dir", type=Path, default=None, help="Путь к клону SeedVR (или SEEDVR_DIR)")
    parser.add_argument("--res_h", type=int, default=720, help="[SeedVR2] Целевая высота (по умолч. 720)")
    parser.add_argument("--res_w", type=int, default=1280, help="[SeedVR2] Целевая ширина (по умолч. 1280)")
    parser.add_argument("--num_gpus", type=int, default=1, help="[SeedVR2] Число GPU для torchrun")
    parser.add_argument("--sp_size", type=int, default=None, help="[SeedVR2] Sequence parallel size")
    parser.add_argument("--seed", type=int, default=42, help="[SeedVR2] Random seed")
    parser.add_argument("--no_audio", action="store_true", help="Не подмешивать/не сохранять аудио")
    # Real-ESRGAN
    parser.add_argument("--realesrgan_dir", type=Path, default=None, help="Путь к клону Real-ESRGAN (или REALESRGAN_DIR)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="realesr-general-x4v3",
        help="[Real-ESRGAN] Модель: realesr-general-x4v3 | RealESRGAN_x4plus | RealESRGAN_x4plus_anime_6B | realesr-animevideov3",
    )
    parser.add_argument("--outscale", type=float, default=2, help="[Real-ESRGAN] Масштаб 2 или 4 (по умолч. 2)")
    parser.add_argument("--tile", type=int, default=0, help="[Real-ESRGAN] Tile size (0 = без тайлов)")
    parser.add_argument("--face_enhance", action="store_true", help="[Real-ESRGAN] GFPGAN для лиц")
    parser.add_argument("--fp32", action="store_true", help="[Real-ESRGAN] FP32 вместо FP16")
    args = parser.parse_args()

    if args.backend == "realesrgan":
        realesrgan_dir = args.realesrgan_dir or os.environ.get("REALESRGAN_DIR")
        if not realesrgan_dir:
            realesrgan_dir = Path("Real-ESRGAN")
        realesrgan_dir = Path(realesrgan_dir)
        if not realesrgan_dir.is_absolute():
            realesrgan_dir = Path.cwd() / realesrgan_dir
        return run_realesrgan_upscale(
            args.input,
            args.output,
            realesrgan_dir=realesrgan_dir,
            model_name=args.model_name,
            outscale=args.outscale,
            tile=args.tile,
            face_enhance=args.face_enhance,
            fp32=args.fp32,
        )

    seedvr_dir = args.seedvr_dir or os.environ.get("SEEDVR_DIR")
    if not seedvr_dir:
        seedvr_dir = Path("SeedVR")
    seedvr_dir = Path(seedvr_dir)
    if not seedvr_dir.is_absolute():
        seedvr_dir = Path.cwd() / seedvr_dir
    return run_upscale(
        args.input,
        args.output,
        seedvr_dir=seedvr_dir,
        res_h=args.res_h,
        res_w=args.res_w,
        num_gpus=args.num_gpus,
        sp_size=args.sp_size,
        seed=args.seed,
        keep_audio=not args.no_audio,
    )


if __name__ == "__main__":
    sys.exit(main())
