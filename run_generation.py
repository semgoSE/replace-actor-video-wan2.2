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
from pathlib import Path


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
    offload_model: bool | None = None,
    multi_gpu: bool = False,
    nproc_per_node: int = 8,
) -> int:
    """
    Запускает generate.py из Wan2.2 (task animate-14B).
    Возвращает код возврата процесса (0 = успех).
    """
    generate_py = wan22_dir / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(
            f"generate.py не найден: {generate_py}\n"
            "Клонируйте репозиторий: git clone https://github.com/Wan-Video/Wan2.2.git"
        )
    if not src_root_path.exists():
        raise FileNotFoundError(f"Папка с данными препроцессинга не найдена: {src_root_path}")

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
    if offload_model is not None:
        cmd.extend(["--offload_model", "True" if offload_model else "False"])
    if replace_flag:
        cmd.append("--replace_flag")
    if use_relighting_lora:
        cmd.append("--use_relighting_lora")

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
        if offload_model is not None:
            cmd.extend(["--offload_model", "True" if offload_model else "False"])
        if replace_flag:
            cmd.append("--replace_flag")
        if use_relighting_lora:
            cmd.append("--use_relighting_lora")

    print("Этап 2: Generation (Wan2.2 generate.py --task animate-14B)")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(wan22_dir))
    if result.returncode == 0 and output_path:
        # generate.py пишет вывод в свою папку; можно скопировать в output_path
        # по умолчанию Wan2.2 пишет в ckpt_dir или текущую папку — см. их README
        print("Генерация завершена. Проверьте вывод в папке Wan2.2 или в логе выше.")
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
    parser.add_argument("--save_file", type=str, default=None, help="Имя выходного файла (иначе Wan2.2 сгенерирует сам)")
    parser.add_argument("--no_offload", action="store_true", help="Не выгружать модель на CPU (быстрее, нужно достаточно VRAM ~24GB+)")
    parser.add_argument("--multi_gpu", action="store_true", help="Запуск на нескольких GPU")
    parser.add_argument("--nproc_per_node", type=int, default=8)
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
        offload_model=False if args.no_offload else None,
        multi_gpu=args.multi_gpu,
        nproc_per_node=args.nproc_per_node,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
