#!/usr/bin/env python3
"""
Web-UI для пайплайна: Preprocess → Generation → Postprocess → Upscale.

Запуск (из корня проекта):
  pip install gradio
  python app_ui.py

Открыть в браузере по выведенному URL (обычно http://127.0.0.1:7860).
Существующие скрипты и запущенные процессы не затрагиваются — UI только вызывает run_*.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Generator

PROJECT_ROOT = Path(__file__).resolve().parent


def _file_path(val) -> str | None:
    """Путь из gr.File: str, list[str] или объект с .name."""
    if val is None:
        return None
    if isinstance(val, str):
        return val if val.strip() else None
    if isinstance(val, list) and val:
        return val[0] if isinstance(val[0], str) else getattr(val[0], "name", None)
    return getattr(val, "name", None)


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    """Запускает команду, возвращает весь вывод по завершении."""
    cwd = cwd or PROJECT_ROOT
    out: list[str] = []
    for chunk in _run_cmd_stream(cmd, cwd):
        out = [chunk]
    return out[0] if out else ""


def _run_cmd_stream(cmd: list[str], cwd: Path | None = None) -> Generator[str, None, None]:
    """Запускает команду и по мере появления вывода выдаёт накопленный лог (realtime)."""
    cwd = cwd or PROJECT_ROOT
    lines: list[str] = []
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    yield "Запуск: " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else "") + "\n\n"
    for line in iter(p.stdout.readline, ""):
        lines.append(line.rstrip())
        yield "\n".join(lines)
    p.wait()
    lines.append(f"\n[Статус: завершено, exit code {p.returncode}]")
    yield "\n".join(lines)


def run_preprocess_ui(
    video_path: str | None,
    refer_path: str | None,
    save_path: str,
    resolution: str,
    fps: int,
    replace: bool,
    wan22_dir: str,
    ckpt_dir: str,
) -> Generator[str, None, None]:
    video_path = _file_path(video_path)
    refer_path = _file_path(refer_path)
    if not video_path or not Path(video_path).exists():
        yield "Укажите существующий файл видео."
        return
    if not refer_path or not Path(refer_path).exists():
        yield "Укажите существующий референс (изображение)."
        return
    save_path = save_path.strip() or "process_results"
    resolution = resolution.strip() or "1280 720"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_preprocessing.py"),
        "--video_path", video_path,
        "--refer_path", refer_path,
        "--save_path", save_path,
        "--resolution", resolution,
        "--fps", str(fps),
        "--wan22_dir", wan22_dir or "Wan2.2",
        "--ckpt_dir", ckpt_dir or "Wan2.2-Animate-14B",
    ]
    if replace:
        cmd.append("--replace")
    else:
        cmd.append("--retarget")
    yield from _run_cmd_stream(cmd)


def run_generation_ui(
    src_root_path: str,
    save_file: str,
    prompt: str | None,
    sample_steps: int | None,
    sample_guide_scale: float | None,
    sample_shift: float | None,
    no_offload: bool,
    replace: bool,
    seed: int,
    wan22_dir: str,
    ckpt_dir: str,
) -> Generator[str, None, None]:
    src = (src_root_path or "").strip()
    if not src or not Path(src).exists():
        yield "Укажите папку с результатом препроцессинга (save_path из этапа 1)."
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_generation.py"),
        "--src_root_path", src,
        "--wan22_dir", wan22_dir or "Wan2.2",
        "--ckpt_dir", ckpt_dir or "Wan2.2-Animate-14B",
        "--seed", str(seed),
    ]
    if save_file:
        cmd.extend(["--save_file", save_file.strip()])
    if prompt:
        cmd.extend(["--prompt", prompt])
    if sample_steps is not None:
        cmd.extend(["--sample_steps", str(sample_steps)])
    if sample_guide_scale is not None:
        cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
    if sample_shift is not None:
        cmd.extend(["--sample_shift", str(sample_shift)])
    if no_offload:
        cmd.append("--no_offload")
    if replace:
        cmd.append("--replace")
    yield from _run_cmd_stream(cmd)


def run_postprocess_ui(
    input_path: str | None,
    output_path: str,
    target_fps: int,
    backend: str,
    no_audio: bool,
) -> Generator[str, None, None]:
    if not input_path or not Path(input_path).exists():
        yield "Укажите существующий входной видеофайл."
        return
    output_path = (output_path or "").strip()
    if not output_path:
        yield "Укажите путь для выходного видео."
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_postprocess.py"),
        "--input", input_path,
        "--output", output_path,
        "--target_fps", str(target_fps),
        "--backend", backend,
    ]
    if no_audio:
        cmd.append("--no_audio")
    yield from _run_cmd_stream(cmd)


def run_upscale_ui(
    input_path: str | None,
    output_path: str,
    backend: str,
    res_h: int,
    res_w: int,
    outscale: float,
    model_name: str,
    no_audio: bool,
) -> Generator[str, None, None]:
    if not input_path or not Path(input_path).exists():
        yield "Укажите существующий входной видеофайл."
        return
    output_path = (output_path or "").strip()
    if not output_path:
        yield "Укажите путь для выходного видео."
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_upscale.py"),
        "--input", input_path,
        "--output", output_path,
        "--backend", backend,
    ]
    if backend == "seedvr2":
        cmd.extend(["--res_h", str(res_h), "--res_w", str(res_w)])
    else:
        cmd.extend(["--outscale", str(outscale), "--model_name", model_name])
    if no_audio:
        cmd.append("--no_audio")
    yield from _run_cmd_stream(cmd)


def run_refine_ui(
    input_path: str | None,
    refer_path: str | None,
    save_path: str,
    save_file: str,
    resolution: str,
    fps: int,
    prompt: str | None,
    sample_steps: int | None,
    sample_guide_scale: float | None,
    sample_shift: float | None,
    no_offload: bool,
    seed: int,
    wan22_dir: str,
    ckpt_dir: str,
) -> Generator[str, None, None]:
    """Refine: второй проход replace по видео для уменьшения артефактов."""
    input_path = _file_path(input_path)
    refer_path = _file_path(refer_path)
    if not input_path or not Path(input_path).exists():
        yield "Укажите видео для рефайна (результат первого replace)."
        return
    if not refer_path or not Path(refer_path).exists():
        yield "Укажите референс (тот же, что в первом replace)."
        return
    save_path = (save_path or "refine_process").strip()
    save_file = (save_file or "refined.mp4").strip()
    resolution = (resolution or "1280 720").strip()
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_refine.py"),
        "--input", input_path,
        "--refer_path", refer_path,
        "--save_path", save_path,
        "--save_file", save_file,
        "--resolution", resolution,
        "--fps", str(int(fps) if fps is not None else 16),
        "--seed", str(int(seed) if seed is not None else 42),
        "--wan22_dir", wan22_dir or "Wan2.2",
        "--ckpt_dir", ckpt_dir or "Wan2.2-Animate-14B",
    ]
    if prompt:
        cmd.extend(["--prompt", prompt])
    if sample_steps is not None:
        cmd.extend(["--sample_steps", str(sample_steps)])
    if sample_guide_scale is not None:
        cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
    if sample_shift is not None:
        cmd.extend(["--sample_shift", str(sample_shift)])
    if no_offload:
        cmd.append("--no_offload")
    yield from _run_cmd_stream(cmd)


def main():
    try:
        import gradio as gr
    except ImportError:
        print("Установите Gradio: pip install gradio", file=sys.stderr)
        sys.exit(1)

    with gr.Blocks(title="Wan2.2 Pipeline", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Wan2.2 Pipeline — Preprocess → Generation → Refine (опц.) → Postprocess → Upscale")

        with gr.Tabs():
            # --- Preprocess ---
            with gr.Tab("1. Preprocess"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.File(label="Видео (поза/движение)", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        refer_in = gr.File(label="Референс (фото персонажа)", file_types=["image"])
                        save_path = gr.Textbox(label="Папка результата (save_path)", value="process_results")
                        resolution = gr.Textbox(label="Разрешение (W H)", value="1280 720")
                        fps = gr.Number(label="FPS", value=16, precision=0)
                        replace_pre = gr.Checkbox(label="Режим Replace (замена актёра)", value=False)
                        wan22_dir = gr.Textbox(label="Wan2.2 (папка)", value="Wan2.2")
                        ckpt_dir = gr.Textbox(label="Чекпоинт (папка)", value="Wan2.2-Animate-14B")
                        btn_pre = gr.Button("Запустить Preprocess")
                    with gr.Column(scale=1):
                        log_pre = gr.Textbox(label="Лог", lines=20, max_lines=40)
                btn_pre.click(
                    fn=lambda v, r, s, res, f, rep, w, c: run_preprocess_ui(
                        _file_path(v), _file_path(r), s, res, int(f) if f is not None else 16, rep, w, c,
                    ),
                    inputs=[video_in, refer_in, save_path, resolution, fps, replace_pre, wan22_dir, ckpt_dir],
                    outputs=[log_pre],
                )

            # --- Generation ---
            with gr.Tab("2. Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        src_root = gr.Textbox(label="Папка препроцессинга (src_root_path)", value="process_results")
                        save_file = gr.Textbox(label="Имя выходного файла (save_file)", value="output.mp4")
                        prompt = gr.Textbox(label="Промпт (пусто = дефолт Wan2.2)")
                        sample_steps = gr.Number(label="Steps", value=30, precision=0)
                        sample_guide_scale = gr.Number(label="CFG", value=2.5)
                        sample_shift = gr.Number(label="Shift", value=5)
                        no_offload = gr.Checkbox(label="Не выгружать модель (--no_offload, быстрее)", value=False)
                        replace_gen = gr.Checkbox(label="Replace", value=False)
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        wan22_dir_gen = gr.Textbox(label="Wan2.2", value="Wan2.2")
                        ckpt_dir_gen = gr.Textbox(label="Чекпоинт", value="Wan2.2-Animate-14B")
                        btn_gen = gr.Button("Запустить Generation")
                    with gr.Column(scale=1):
                        log_gen = gr.Textbox(label="Лог", lines=20, max_lines=40)
                btn_gen.click(
                    fn=lambda src, sf, p, st, cfg, sh, no, rep, se, w, c: run_generation_ui(
                        src, sf, p or None,
                        int(st) if st is not None else None,
                        float(cfg) if cfg is not None else None,
                        float(sh) if sh is not None else None,
                        no, rep, int(se) if se is not None else 42, w, c,
                    ),
                    inputs=[src_root, save_file, prompt, sample_steps, sample_guide_scale, sample_shift,
                            no_offload, replace_gen, seed, wan22_dir_gen, ckpt_dir_gen],
                    outputs=[log_gen],
                )

            # --- Refine (второй replace для артефактов) ---
            with gr.Tab("Refine"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_in = gr.File(label="Видео для рефайна (результат первого replace)", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        ref_refer = gr.File(label="Референс (тот же)", file_types=["image"])
                        ref_save_path = gr.Textbox(label="Папка препроцессинга refine", value="refine_process")
                        ref_save_file = gr.Textbox(label="Имя выходного файла", value="refined.mp4")
                        ref_resolution = gr.Textbox(label="Разрешение (W H)", value="1280 720")
                        ref_fps = gr.Number(label="FPS", value=16, precision=0)
                        ref_prompt = gr.Textbox(label="Промпт (пусто = дефолт)")
                        ref_steps = gr.Number(label="Steps", value=30, precision=0)
                        ref_cfg = gr.Number(label="CFG", value=2.5)
                        ref_shift = gr.Number(label="Shift", value=5)
                        ref_no_offload = gr.Checkbox(label="Не выгружать модель", value=False)
                        ref_seed = gr.Number(label="Seed", value=42, precision=0)
                        ref_wan22 = gr.Textbox(label="Wan2.2", value="Wan2.2")
                        ref_ckpt = gr.Textbox(label="Чекпоинт", value="Wan2.2-Animate-14B")
                        btn_ref = gr.Button("Запустить Refine")
                    with gr.Column(scale=1):
                        log_ref = gr.Textbox(label="Лог", lines=20, max_lines=40)
                btn_ref.click(
                    fn=lambda i, r, sp, sf, res, f, p, st, cfg, sh, no, se, w, c: run_refine_ui(
                        _file_path(i), _file_path(r), sp, sf, res, f, p or None,
                        int(st) if st is not None else None,
                        float(cfg) if cfg is not None else None,
                        float(sh) if sh is not None else None,
                        no, se, w, c,
                    ),
                    inputs=[ref_in, ref_refer, ref_save_path, ref_save_file, ref_resolution, ref_fps,
                            ref_prompt, ref_steps, ref_cfg, ref_shift, ref_no_offload, ref_seed, ref_wan22, ref_ckpt],
                    outputs=[log_ref],
                )

            # --- Postprocess (RIFE / ffmpeg) ---
            with gr.Tab("3. Postprocess (FPS)")
                with gr.Row():
                    with gr.Column(scale=1):
                        post_in = gr.File(label="Входное видео", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        post_out = gr.Textbox(label="Выходное видео (путь)")
                        target_fps = gr.Number(label="Целевой FPS", value=30, precision=0)
                        backend_post = gr.Radio(label="Бэкенд", choices=["vulkan", "cuda", "ffmpeg"], value="vulkan")
                        no_audio_post = gr.Checkbox(label="Без аудио", value=False)
                        btn_post = gr.Button("Запустить Postprocess")
                    with gr.Column(scale=1):
                        log_post = gr.Textbox(label="Лог", lines=20, max_lines=40)
                btn_post.click(
                    fn=lambda i, o, f, b, na: run_postprocess_ui(
                        _file_path(i), o, int(f) if f is not None else 30, b, na,
                    ),
                    inputs=[post_in, post_out, target_fps, backend_post, no_audio_post],
                    outputs=[log_post],
                )

            # --- Upscale ---
            with gr.Tab("4. Upscale"):
                with gr.Row():
                    with gr.Column(scale=1):
                        up_in = gr.File(label="Входное видео", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        up_out = gr.Textbox(label="Выходное видео (путь)")
                        backend_up = gr.Radio(label="Бэкенд", choices=["seedvr2", "realesrgan"], value="realesrgan")
                        res_h = gr.Number(label="[SeedVR2] Высота", value=720, precision=0)
                        res_w = gr.Number(label="[SeedVR2] Ширина", value=1280, precision=0)
                        outscale = gr.Number(label="[Real-ESRGAN] Масштаб (2 или 4)", value=2)
                        model_name = gr.Dropdown(
                            label="[Real-ESRGAN] Модель",
                            choices=["realesr-general-x4v3", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-animevideov3"],
                            value="realesr-general-x4v3",
                        )
                        no_audio_up = gr.Checkbox(label="Без аудио", value=False)
                        btn_up = gr.Button("Запустить Upscale")
                    with gr.Column(scale=1):
                        log_up = gr.Textbox(label="Лог", lines=20, max_lines=40)
                btn_up.click(
                    fn=lambda i, o, b, rh, rw, sc, mn, na: run_upscale_ui(
                        _file_path(i), o, b,
                        int(rh) if rh is not None else 720,
                        int(rw) if rw is not None else 1280,
                        float(sc) if sc is not None else 2,
                        mn or "realesr-general-x4v3",
                        na,
                    ),
                    inputs=[up_in, up_out, backend_up, res_h, res_w, outscale, model_name, no_audio_up],
                    outputs=[log_up],
                )

    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
