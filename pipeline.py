#!/usr/bin/env python3
"""
Полный пайплайн Wan2.2-Animate в два этапа (чисто Python, без ComfyUI).

  Этап 1 — Preprocessing: видео + референс → process_results
  Этап 2 — Generation: process_results → выходное видео

Запуск:
  python pipeline.py --video_path dance.mp4 --refer_path face.jpg
  python pipeline.py --video_path v.mp4 --refer_path r.jpg --replace  # режим замены актёра
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import PipelineConfig

from run_preprocessing import run_preprocessing
from run_generation import run_generation


def run_pipeline(config: PipelineConfig) -> int:
    """Выполняет этап 1, затем этап 2. Возвращает 0 при успехе."""
    print("=== Pipeline Wan2.2-Animate (2 этапа) ===\n")

    ret = run_preprocessing(
        wan22_dir=config.wan22_dir,
        ckpt_dir=config.ckpt_dir,
        video_path=config.video_path,
        refer_path=config.refer_path,
        save_path=config.save_path,
        resolution_area=config.resolution_area,
        retarget_flag=config.retarget_flag,
        replace_flag=config.replace_flag,
        use_flux=config.use_flux,
        iterations=config.iterations,
        k=config.k,
        w_len=config.w_len,
        h_len=config.h_len,
    )
    if ret != 0:
        print("Этап 1 (Preprocessing) завершился с ошибкой.")
        return ret

    print()
    ret = run_generation(
        wan22_dir=config.wan22_dir,
        ckpt_dir=config.ckpt_dir,
        src_root_path=config.save_path,
        output_path=config.output_path,
        refert_num=config.refert_num,
        replace_flag=config.replace_flag,
        use_relighting_lora=config.use_relighting_lora,
        seed=config.seed,
        multi_gpu=config.multi_gpu,
        nproc_per_node=config.nproc_per_node,
    )
    if ret != 0:
        print("Этап 2 (Generation) завершился с ошибкой.")
        return ret

    print("\nPipeline завершён успешно.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Pipeline Wan2.2-Animate: preprocessing + generation")
    parser.add_argument("--video_path", type=Path, required=True, help="Входное видео")
    parser.add_argument("--refer_path", type=Path, required=True, help="Референсное изображение")
    parser.add_argument("--wan22_dir", type=Path, default=Path("Wan2.2"))
    parser.add_argument("--ckpt_dir", type=Path, default=Path("Wan2.2-Animate-14B"))
    parser.add_argument("--save_path", type=Path, default=Path("process_results"))
    parser.add_argument("--output_path", type=Path, default=Path("output_wanimate.mp4"))
    parser.add_argument("--resolution", type=str, default="1280 720")
    parser.add_argument("--retarget", action="store_true", default=True, help="Animate: персонаж повторяет движение")
    parser.add_argument("--replace", action="store_true", help="Replace: замена актёра в видео")
    parser.add_argument("--use_flux", action="store_true")
    parser.add_argument("--use_relighting_lora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=8)
    args = parser.parse_args()

    res = [int(x) for x in args.resolution.split()]
    resolution_area = (res[0], res[1]) if len(res) >= 2 else (1280, 720)

    config = PipelineConfig(
        wan22_dir=args.wan22_dir,
        ckpt_dir=args.ckpt_dir,
        video_path=args.video_path,
        refer_path=args.refer_path,
        save_path=args.save_path,
        output_path=args.output_path,
        resolution_area=resolution_area,
        retarget_flag=args.retarget and not args.replace,
        replace_flag=args.replace,
        use_flux=args.use_flux,
        use_relighting_lora=args.use_relighting_lora,
        seed=args.seed,
        multi_gpu=args.multi_gpu,
        nproc_per_node=args.nproc_per_node,
    )
    sys.exit(run_pipeline(config))


if __name__ == "__main__":
    main()
