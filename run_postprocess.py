#!/usr/bin/env python3
"""
Этап 3 (опционально): Post-processing — повышение FPS через RIFE.

Принимает видео после animate/replace (например 16 fps), дорисовывает кадры
и сохраняет с целевым FPS (по умолчанию 30).

Режимы (--backend):
  vulkan (по умолч.): rife-ncnn-vulkan, нужен Vulkan.
  cuda: Practical-RIFE (PyTorch/CUDA), для H100/A100 без Vulkan.
  ffmpeg: minterpolate, без RIFE.

Использование:
  python run_postprocess.py --input video.mp4 --output out.mp4 --target_fps 30
  python run_postprocess.py --input video.mp4 --output out.mp4 --backend cuda --rife_pytorch_dir /path/to/Practical-RIFE
  python run_postprocess.py --input video.mp4 --output out.mp4 --backend ffmpeg
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
    """Возвращает FPS входного видео через ffprobe или None."""
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


def find_rife_bin(rife_bin: Path | None) -> Path | None:
    """Ищет бинарник rife-ncnn-vulkan в PATH или рядом со скриптом."""
    names = ["rife-ncnn-vulkan", "rife-ncnn-vulkan.exe"]
    if rife_bin is not None and rife_bin.exists():
        return Path(rife_bin)
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    env_bin = os.environ.get("RIFE_NCNN_VULKAN")
    if env_bin and Path(env_bin).exists():
        return Path(env_bin)
    return None


def run_rife_interpolate(
    input_dir: Path,
    output_dir: Path,
    rife_bin: Path,
) -> int:
    """Запускает rife-ncnn-vulkan: input_dir -> output_dir (2x кадров по умолчанию)."""
    # -f "%08d.png" чтобы кадры лежали прямо в output_dir, не в output_dir/ext
    cmd = [
        str(rife_bin), "-i", str(input_dir), "-o", str(output_dir),
        "-f", "%08d.png",
    ]
    r = subprocess.run(cmd)
    return r.returncode


def run_postprocess_rife(
    input_video: Path,
    output_video: Path,
    target_fps: int = 30,
    rife_bin: Path | None = None,
    keep_audio: bool = True,
) -> int:
    """
    Постобработка через RIFE: извлекаем кадры -> RIFE 2x -> собираем видео с target_fps.
    """
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    if not input_video.exists():
        print(f"Ошибка: файл не найден: {input_video}", file=sys.stderr)
        return 1

    rife = find_rife_bin(Path(rife_bin) if rife_bin else None)
    if rife is None:
        print(
            "RIFE (rife-ncnn-vulkan) не найден. Установите: https://github.com/nihui/rife-ncnn-vulkan/releases\n"
            "На серверных GPU (H100, A100) Vulkan может не работать — используйте --fallback_ffmpeg.\n"
            "Либо укажите --rife_bin /путь/к/rife-ncnn-vulkan.",
            file=sys.stderr,
        )
        return 1

    input_fps = get_input_fps(input_video)
    if input_fps is None:
        input_fps = 16.0
        print(f"Не удалось определить FPS, считаем {input_fps}", file=sys.stderr)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="rife_post_"))
    try:
        frames_in = tmp / "in"
        frames_out = tmp / "out"
        frames_in.mkdir()
        frames_out.mkdir()
        audio_file = tmp / "audio.m4a"

        # 1) Извлечь кадры и аудио
        print("Извлечение кадров и аудио...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_video),
                "-vn", "-acodec", "copy", str(audio_file),
            ],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_video),
                str(frames_in / "%08d.png"),
            ],
            capture_output=True,
            check=True,
        )
        num_input = len(list(frames_in.glob("*.png")))
        if num_input == 0:
            print("Не удалось извлечь кадры.", file=sys.stderr)
            return 1

        # 2) RIFE 2x: между каждой парой кадров один новый
        print(f"RIFE интерполяция 2x ({num_input} кадров -> ~2x)...")
        ret = run_rife_interpolate(frames_in, frames_out, rife)
        if ret != 0:
            print("Ошибка RIFE.", file=sys.stderr)
            return ret

        out_frames_dir = frames_out
        num_out = len(list(frames_out.glob("*.png")))
        if num_out == 0:
            # часть сборок rife-ncnn-vulkan пишет в подпапку ext
            ext_dir = frames_out / "ext"
            if ext_dir.exists():
                out_frames_dir = ext_dir
                num_out = len(list(ext_dir.glob("*.png")))
        if num_out == 0:
            print("RIFE не создал кадры.", file=sys.stderr)
            return 1

        # 3) Собрать видео: интерполированные кадры при 2*input_fps, затем пересчитать в target_fps
        interpolated_fps = 2.0 * input_fps
        print(f"Сборка видео {num_out} кадров -> {target_fps} fps...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(interpolated_fps),
            "-i", str(out_frames_dir / "%08d.png"),
        ]
        if keep_audio and audio_file.exists():
            cmd += ["-i", str(audio_file), "-c:a", "copy"]
        cmd += [
            "-r", str(target_fps),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(output_video),
        ]
        subprocess.run(cmd, check=True)
        print("Готово:", output_video)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_postprocess_rife_cuda(
    input_video: Path,
    output_video: Path,
    target_fps: int = 30,
    rife_pytorch_dir: Path | None = None,
    keep_audio: bool = True,
) -> int:
    """
    Постобработка через RIFE на PyTorch/CUDA (Practical-RIFE).
    Нужен клон https://github.com/hzwer/Practical-RIFE и скачанные веса в train_log/.
    """
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    if not input_video.exists():
        print(f"Ошибка: файл не найден: {input_video}", file=sys.stderr)
        return 1

    rife_dir = rife_pytorch_dir or os.environ.get("RIFE_PYTORCH_DIR") or Path("Practical-RIFE")
    rife_dir = Path(rife_dir).resolve()
    inference_script = rife_dir / "inference_video.py"
    if not inference_script.exists():
        print(
            "Practical-RIFE не найден. Клонируйте и установите:\n"
            "  git clone https://github.com/hzwer/Practical-RIFE\n"
            "  cd Practical-RIFE && pip install -r requirements.txt\n"
            "  # скачайте веса в train_log/ (см. их README)\n"
            "Укажите путь: --rife_pytorch_dir ./Practical-RIFE или RIFE_PYTORCH_DIR (по умолчанию — ./Practical-RIFE в папке проекта).",
            file=sys.stderr,
        )
        return 1

    output_video.parent.mkdir(parents=True, exist_ok=True)
    # При --fps Practical-RIFE не подмешивает аудио — делаем это сами
    tmp_out = output_video.parent / (output_video.stem + "_rife_tmp" + output_video.suffix)
    cmd = [
        sys.executable,
        str(inference_script),
        "--video", str(input_video),
        "--output", str(tmp_out),
        "--exp", "1",
        "--fps", str(target_fps),
    ]
    print("RIFE (PyTorch/CUDA) интерполяция 2x ->", target_fps, "fps...")
    r = subprocess.run(cmd, cwd=str(rife_dir))
    if r.returncode != 0:
        return r.returncode
    if keep_audio and tmp_out.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tmp_out), "-i", str(input_video),
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
                str(output_video),
            ],
            capture_output=True,
            check=True,
        )
        tmp_out.unlink(missing_ok=True)
    else:
        if tmp_out.exists():
            tmp_out.rename(output_video)
    print("Готово:", output_video)
    return 0


def run_postprocess_ffmpeg(
    input_video: Path,
    output_video: Path,
    target_fps: int = 30,
    keep_audio: bool = True,
) -> int:
    """Fallback: ffmpeg minterpolate (без RIFE, качество хуже)."""
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    # minterpolate=fps=30:mi_mode=mci (motion compensated)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_video),
        "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:a", "copy" if keep_audio else "-an",
        str(output_video),
    ]
    print("Постобработка (ffmpeg minterpolate)...")
    r = subprocess.run(cmd)
    if r.returncode == 0:
        print("Готово:", output_video)
    return r.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Постобработка: повышение FPS (RIFE или ffmpeg minterpolate)"
    )
    parser.add_argument("--input", type=Path, required=True, help="Входное видео (например 16 fps)")
    parser.add_argument("--output", type=Path, required=True, help="Выходное видео (например 30 fps)")
    parser.add_argument("--target_fps", type=int, default=30, help="Целевой FPS (по умолч. 30)")
    parser.add_argument("--rife_bin", type=Path, default=None, help="Путь к rife-ncnn-vulkan")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vulkan", "cuda", "ffmpeg"],
        default="vulkan",
        help="vulkan=rife-ncnn-vulkan, cuda=Practical-RIFE (PyTorch), ffmpeg=minterpolate",
    )
    parser.add_argument(
        "--rife_pytorch_dir",
        type=Path,
        default=None,
        help="Путь к клону Practical-RIFE (для --backend cuda). По умолч. ./Practical-RIFE в папке проекта.",
    )
    parser.add_argument(
        "--fallback_ffmpeg",
        action="store_true",
        help="То же что --backend ffmpeg (только ffmpeg minterpolate)",
    )
    parser.add_argument("--no_audio", action="store_true", help="Не копировать аудио")
    args = parser.parse_args()

    backend = "ffmpeg" if args.fallback_ffmpeg else args.backend
    if backend == "ffmpeg":
        return run_postprocess_ffmpeg(
            args.input,
            args.output,
            target_fps=args.target_fps,
            keep_audio=not args.no_audio,
        )
    if backend == "cuda":
        return run_postprocess_rife_cuda(
            args.input,
            args.output,
            target_fps=args.target_fps,
            rife_pytorch_dir=args.rife_pytorch_dir,
            keep_audio=not args.no_audio,
        )
    return run_postprocess_rife(
        args.input,
        args.output,
        target_fps=args.target_fps,
        rife_bin=args.rife_bin,
        keep_audio=not args.no_audio,
    )


if __name__ == "__main__":
    sys.exit(main())
