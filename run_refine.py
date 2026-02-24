#!/usr/bin/env python3
"""
Этап Refine (video-to-video): второй проход replace для уменьшения артефактов.

Вход: видео после первого replace (или после generation) — часто с артефактами
(остатки одежды, границы маски и т.п.).
Выход: то же видео, прогнанное через preprocess + generation (replace) ещё раз —
модель перерисовывает персонажа и может убрать артефакты.

Использование:
  python run_refine.py --input out_replace.mp4 --refer_path ref.jpg --save_file out_refined.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_preprocessing import run_preprocessing
from run_generation import run_generation


def run_refine(
    input_video: Path,
    refer_path: Path,
    save_path: Path,
    save_file: str,
    wan22_dir: Path,
    ckpt_dir: Path,
    resolution: tuple[int, int] = (1280, 720),
    fps: int | None = 16,
    prompt: str | None = None,
    sample_steps: int | None = None,
    sample_guide_scale: float | None = None,
    sample_shift: float | None = None,
    seed: int = 42,
    offload_model: bool | None = None,
    use_relighting_lora: bool = False,
) -> int:
    """
    Refine: preprocess(видео=input, replace) → generation(replace).
    Возвращает 0 при успехе.
    """
    input_video = Path(input_video).resolve()
    refer_path = Path(refer_path).resolve()
    save_path = Path(save_path).resolve()
    wan22_dir = Path(wan22_dir).resolve()
    ckpt_dir = Path(ckpt_dir).resolve()

    if not input_video.exists():
        print(f"Ошибка: видео не найдено: {input_video}", file=sys.stderr)
        return 1
    if not refer_path.exists():
        print(f"Ошибка: референс не найден: {refer_path}", file=sys.stderr)
        return 1

    save_path.mkdir(parents=True, exist_ok=True)
    print("=== Refine (второй проход replace для уменьшения артефактов) ===\n")

    ret = run_preprocessing(
        wan22_dir=wan22_dir,
        ckpt_dir=ckpt_dir,
        video_path=input_video,
        refer_path=refer_path,
        save_path=save_path,
        resolution_area=resolution,
        fps=fps,
        retarget_flag=False,
        replace_flag=True,
        use_flux=False,
    )
    if ret != 0:
        print("Refine: препроцессинг завершился с ошибкой.", file=sys.stderr)
        return ret

    ret = run_generation(
        wan22_dir=wan22_dir,
        ckpt_dir=ckpt_dir,
        src_root_path=save_path,
        output_path=Path(save_file).resolve() if save_file else Path("refined.mp4"),
        refert_num=1,
        replace_flag=True,
        use_relighting_lora=use_relighting_lora,
        seed=seed,
        prompt=prompt,
        sample_steps=sample_steps,
        sample_guide_scale=sample_guide_scale,
        sample_shift=sample_shift,
        save_file=save_file,
        offload_model=offload_model,
    )
    if ret != 0:
        print("Refine: генерация завершилась с ошибкой.", file=sys.stderr)
        return ret

    print("\nRefine завершён. Результат:", save_file)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Refine: второй проход replace по видео для уменьшения артефактов (остатки одежды и т.п.)"
    )
    parser.add_argument("--input", type=Path, required=True, help="Входное видео (результат первого replace)")
    parser.add_argument("--refer_path", type=Path, required=True, help="Референс (тот же, что в первом replace)")
    parser.add_argument("--save_path", type=Path, default=Path("refine_process"), help="Папка для препроцессинга refine")
    parser.add_argument("--save_file", type=str, default="refined.mp4", help="Имя выходного файла")
    parser.add_argument("--wan22_dir", type=Path, default=Path("Wan2.2"))
    parser.add_argument("--ckpt_dir", type=Path, default=Path("Wan2.2-Animate-14B"))
    parser.add_argument("--resolution", type=str, default="1280 720", help="Разрешение W H")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_offload", action="store_true", help="Не выгружать модель (быстрее)")
    parser.add_argument("--use_relighting_lora", action="store_true")
    args = parser.parse_args()

    res = [int(x) for x in args.resolution.split()]
    resolution = (res[0], res[1]) if len(res) >= 2 else (1280, 720)

    return run_refine(
        input_video=args.input,
        refer_path=args.refer_path,
        save_path=args.save_path,
        save_file=args.save_file,
        wan22_dir=args.wan22_dir,
        ckpt_dir=args.ckpt_dir,
        resolution=resolution,
        fps=args.fps,
        prompt=args.prompt,
        sample_steps=args.sample_steps,
        sample_guide_scale=args.sample_guide_scale,
        sample_shift=args.sample_shift,
        seed=args.seed,
        offload_model=False if args.no_offload else None,
        use_relighting_lora=args.use_relighting_lora,
    )


if __name__ == "__main__":
    sys.exit(main())
