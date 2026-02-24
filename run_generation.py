#!/usr/bin/env python3
"""
Этап 2: Generation.

Запускает официальную генерацию Wan2.2 (generate.py --task animate-14B):
- вход: папка process_results с этапа 1 (preprocessing)
- выход: видеофайл

Требуется: Wan2.2 репозиторий и чекпоинт Wan2.2-Animate-14B.

Использование:
  python run_generation.py --src_root_path process_results --output_path out.mp4
  python run_generation.py --config config.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

from preprocess import ensure_unique_path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "out"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm")


def _get_fps_from_video(path: Path) -> float | None:
    """FPS видео через ffprobe (r_frame_rate)."""
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
            return float(num) / float(den) if float(den) else None
        return float(rate)
    except Exception:
        return None


def _get_fps_from_preprocess_folder(src_root_path: Path) -> float | None:
    """FPS первого видео в папке препроцессинга (чтобы выход Generation совпадал с входом)."""
    path = Path(src_root_path)
    if not path.is_dir():
        return None
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in VIDEO_EXTENSIONS:
            fps = _get_fps_from_video(f)
            if fps is not None:
                return fps
    return None


def run_generation(
    wan22_dir: Path,
    ckpt_dir: Path,
    src_root_path: Path,
    output_path: Path,
    refert_num: int = 1,
    replace_flag: bool = False,
    use_relighting_lora: bool = False,
    seed: int = 42,
    prompt: str | None = None,
    sample_steps: int | None = None,
    sample_guide_scale: float | None = None,
    sample_shift: float | None = None,
    frame_num: int | None = None,
    save_file: str | None = None,
    sample_solver: str | None = None,
    offload_model: bool | None = None,
    multi_gpu: bool = False,
    nproc_per_node: int = 8,
    sample_fps: float | None = None,
) -> int:
    """
    Запускает generate.py из Wan2.2 (task animate-14B).
    Возвращает код возврата процесса (0 = успех).
    Если sample_fps не задан, берётся FPS из первого видео в папке препроцессинга.
    """
    generate_py = wan22_dir / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(
            f"generate.py не найден: {generate_py}\n"
            "Клонируйте репозиторий: git clone https://github.com/Wan-Video/Wan2.2.git"
        )
    if not src_root_path.exists():
        raise FileNotFoundError(f"Папка с данными препроцессинга не найдена: {src_root_path}")

    # FPS выхода: явный sample_fps или из первого видео в папке препроцессинга
    if sample_fps is None:
        sample_fps = _get_fps_from_preprocess_folder(src_root_path)
        if sample_fps is not None:
            print(f"Generation: FPS выходного видео задан от входа (препроцессинг): {sample_fps}")

    # Если файл уже есть — сохраняем как _1, _2, ...
    if save_file:
        out_full = (wan22_dir / save_file).resolve()
        save_file = ensure_unique_path(out_full).name

    # при cwd=wan22_dir скрипт — просто generate.py
    gen_script = "generate.py"
    cmd = [
        sys.executable,
        gen_script,
        "--task", "animate-14B",
        "--ckpt_dir", str(ckpt_dir.resolve()),
        "--src_root_path", str(src_root_path.resolve()),
        "--refert_num", str(refert_num),
        "--base_seed", str(seed),
    ]
    if prompt is not None:
        cmd.extend(["--prompt", prompt])
    if sample_steps is not None:
        cmd.extend(["--sample_steps", str(sample_steps)])
    if sample_guide_scale is not None:
        cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
    if sample_shift is not None:
        cmd.extend(["--sample_shift", str(sample_shift)])
    if frame_num is not None:
        cmd.extend(["--frame_num", str(frame_num)])
    if save_file:
        cmd.extend(["--save_file", str(save_file)])
    if sample_solver is not None:
        cmd.extend(["--sample_solver", sample_solver])
    if offload_model is not None:
        cmd.extend(["--offload_model", "True" if offload_model else "False"])
    if replace_flag:
        cmd.append("--replace_flag")
    if use_relighting_lora:
        cmd.append("--use_relighting_lora")
    if sample_fps is not None:
        cmd.extend(["--sample_fps", str(sample_fps)])

    if multi_gpu:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", "1",
            "--nproc_per_node", str(nproc_per_node),
            gen_script,
            "--task", "animate-14B",
            "--ckpt_dir", str(ckpt_dir.resolve()),
            "--src_root_path", str(src_root_path.resolve()),
            "--refert_num", str(refert_num),
            "--base_seed", str(seed),
            "--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(nproc_per_node),
        ]
        if prompt is not None:
            cmd.extend(["--prompt", prompt])
        if sample_steps is not None:
            cmd.extend(["--sample_steps", str(sample_steps)])
        if sample_guide_scale is not None:
            cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
        if sample_shift is not None:
            cmd.extend(["--sample_shift", str(sample_shift)])
        if frame_num is not None:
            cmd.extend(["--frame_num", str(frame_num)])
        if save_file:
            cmd.extend(["--save_file", str(save_file)])
        if sample_solver is not None:
            cmd.extend(["--sample_solver", sample_solver])
        if offload_model is not None:
            cmd.extend(["--offload_model", "True" if offload_model else "False"])
        if replace_flag:
            cmd.append("--replace_flag")
        if use_relighting_lora:
            cmd.append("--use_relighting_lora")
        if sample_fps is not None:
            cmd.extend(["--sample_fps", str(sample_fps)])

    print("Этап 2: Generation (Wan2.2 generate.py --task animate-14B)")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(wan22_dir))
    if result.returncode == 0 and save_file:
        # Копируем результат в папку out/ в корне проекта
        src = wan22_dir / save_file
        if not src.exists():
            src = wan22_dir / "out" / save_file
        if src.exists():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            dest = ensure_unique_path(OUTPUT_DIR / save_file)
            shutil.copy2(src, dest)
            print("Генерация завершена. Результат сохранён:", dest)
        else:
            print("Генерация завершена. Проверьте вывод в папке Wan2.2 или Wan2.2/out/.")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Этап 2: Generation Wan2.2-Animate")
    parser.add_argument("--wan22_dir", type=Path, default=Path("Wan2.2"), help="Путь к репозиторию Wan2.2")
    parser.add_argument("--ckpt_dir", type=Path, default=Path("Wan2.2-Animate-14B"), help="Путь к чекпоинту")
    parser.add_argument("--src_root_path", type=Path, required=True, help="Папка с результатом препроцессинга (этап 1)")
    parser.add_argument("--output_path", type=Path, default=Path("output_wanimate.mp4"), help="Куда сохранить видео (если скрипт поддерживает)")
    parser.add_argument("--refert_num", type=int, default=1)
    parser.add_argument("--replace", action="store_true", help="Режим replace")
    parser.add_argument("--use_relighting_lora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default=None, help="Текстовый промпт (по умолчанию дефолт Wan2.2: 视频中的人在做动作)")
    parser.add_argument("--sample_steps", type=int, default=None, help="Шаги сэмплинга (дефолт в Wan2.2 для animate ~30–40)")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="CFG / guidance scale (дефолт в Wan2.2 ~2–4.5)")
    parser.add_argument("--sample_shift", type=float, default=None, help="Flow shift (дефолт в Wan2.2 ~3–5)")
    parser.add_argument("--frame_num", type=int, default=None, help="Число кадров (4n+1)")
    parser.add_argument("--save_file", type=str, default=None, help="Имя выходного файла (результат также копируется в out/)")
    parser.add_argument("--sample_solver", type=str, default=None, choices=["unipc", "dpm++"], help="Сэмплер: unipc или dpm++")
    parser.add_argument("--no_offload", action="store_true", help="Не выгружать модель на CPU (быстрее, нужно достаточно VRAM ~24GB+)")
    parser.add_argument("--multi_gpu", action="store_true", help="Запуск на нескольких GPU")
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument("--fps", type=float, default=None, help="FPS выходного видео (по умолчанию — от препроцессинга)")
    args = parser.parse_args()

    ret = run_generation(
        wan22_dir=args.wan22_dir,
        ckpt_dir=args.ckpt_dir,
        src_root_path=args.src_root_path,
        output_path=args.output_path,
        refert_num=args.refert_num,
        replace_flag=args.replace,
        use_relighting_lora=args.use_relighting_lora,
        seed=args.seed,
        prompt=args.prompt,
        sample_steps=args.sample_steps,
        sample_guide_scale=args.sample_guide_scale,
        sample_shift=args.sample_shift,
        frame_num=args.frame_num,
        save_file=args.save_file,
        sample_solver=args.sample_solver,
        offload_model=False if args.no_offload else None,
        multi_gpu=args.multi_gpu,
        nproc_per_node=args.nproc_per_node,
        sample_fps=args.fps,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
