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

import json
import subprocess
import sys
from pathlib import Path
from typing import Generator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# Текущие запущенные процессы по вкладкам (для кнопки «Остановить»)
_current_processes: dict[str, subprocess.Popen] = {}

# Расширения для сканирования на сервере
SERVER_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SERVER_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
SERVER_INPUTS_DIR = PROJECT_ROOT / "inputs"  # папка по умолчанию для референсов и видео
OUTPUT_DIRS = [PROJECT_ROOT / "Wan2.2" / "out", PROJECT_ROOT / "out"]
# Все результаты препроцессинга хранятся в preprocess/<имя>
PREPROCESS_BASE = PROJECT_ROOT / "preprocess"


def _file_path(val) -> str | None:
    """Путь из gr.File: str, list[str] или объект с .name."""
    if val is None:
        return None
    if isinstance(val, str):
        return val if val.strip() else None
    if isinstance(val, list) and val:
        return val[0] if isinstance(val[0], str) else getattr(val[0], "name", None)
    return getattr(val, "name", None)


def _list_server_files(root: str | Path, exts: set[str], recursive: bool = True) -> list[str]:
    """Сканирует папку на сервере, возвращает список путей к файлам с нужными расширениями."""
    root = Path(root or ".").resolve()
    if not root.is_dir():
        root = PROJECT_ROOT
    out: list[str] = []
    try:
        it = root.rglob("*") if recursive else root.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                try:
                    out.append(str(p.resolve()))
                except Exception:
                    pass
    except Exception:
        pass
    out.sort()
    return out


def _resolve_input(server_choice: str | None, uploaded) -> str | None:
    """Приоритет: выбранный с сервера путь, иначе загруженный файл."""
    if server_choice and str(server_choice).strip() and Path(server_choice).exists():
        return str(Path(server_choice).resolve())
    return _file_path(uploaded)


def _list_output_files() -> list[str]:
    """Список путей к выходным видео для вкладки Скачать."""
    out: list[str] = []
    for d in OUTPUT_DIRS:
        if d.exists():
            for p in sorted(d.glob("*.mp4")) + sorted(d.glob("*.webm")):
                out.append(str(p.resolve()))
    return out[:200]


def _list_preprocess_folder_names() -> list[str]:
    """Имена подпапок в preprocess/ (для выбора имени при сохранении препроцессинга)."""
    if not PREPROCESS_BASE.exists():
        return []
    out = [d.name for d in sorted(PREPROCESS_BASE.iterdir()) if d.is_dir()]
    return out


def _list_preprocess_folders() -> list[str]:
    """Полные пути папок препроцессинга для выбора в Generation (preprocess/* + legacy)."""
    out: list[str] = []
    if PREPROCESS_BASE.exists():
        for d in sorted(PREPROCESS_BASE.iterdir()):
            if d.is_dir():
                out.append(str(d.resolve()))
    for legacy in ("process_results", "refine_process"):
        p = PROJECT_ROOT / legacy
        if p.exists() and p.is_dir() and str(p.resolve()) not in out:
            out.append(str(p.resolve()))
    return out


def _list_preprocess_folders_for_thin() -> list[str]:
    """Папки препроцессинга для входа Thin (без _gen, чтобы не проредить уже прореженное)."""
    return [p for p in _list_preprocess_folders() if not Path(p).name.endswith("_gen")]


def _list_sam2_points_files() -> list[str]:
    """Сохранённые файлы точек для SAM2 (preprocess/sam2_points_*.json)."""
    out = []
    if PREPROCESS_BASE.exists():
        for p in sorted(PREPROCESS_BASE.glob("sam2_points_*.json")):
            out.append(str(p.resolve()))
    return out


def _extract_first_frame(video_path: str | None) -> tuple[np.ndarray | None, str]:
    """Извлекает первый кадр из видео. Возвращает (numpy RGB [0,255] или None, сообщение)."""
    if not video_path or not Path(video_path).exists():
        return None, "Укажите видео"
    try:
        from preprocess import load_video_frames
        frames = load_video_frames(video_path, max_frames=1)
        if frames.size == 0:
            return None, "Нет кадров"
        fr = (np.clip(frames[0], 0, 1) * 255).astype(np.uint8)
        if fr.shape[-1] == 4:
            fr = fr[..., :3]
        return fr, f"Кадр 0: {fr.shape[1]}×{fr.shape[0]}"
    except Exception as e:
        return None, str(e)


def _refresh_server_inputs(server_dir: str) -> tuple[list[str], list[str]]:
    """Сканирует папку на сервере + out (Wan2.2/out, out/), возвращает (видео, изображения) для dropdown."""
    root = Path(server_dir or "inputs").resolve()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    videos: list[str] = []
    images: list[str] = []
    if root.exists():
        videos = _list_server_files(root, SERVER_VIDEO_EXTS)
        images = _list_server_files(root, SERVER_IMAGE_EXTS)
    # Добавляем готовые результаты из out — особенно для шагов после первого
    for d in OUTPUT_DIRS:
        if d.exists():
            for p in sorted(d.glob("*.mp4")) + sorted(d.glob("*.webm")) + sorted(d.glob("*.mov")):
                path_str = str(p.resolve())
                if path_str not in videos:
                    videos.append(path_str)
    videos.sort()
    return videos, images


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    """Запускает команду, возвращает весь вывод по завершении."""
    cwd = cwd or PROJECT_ROOT
    out: list[str] = []
    for chunk in _run_cmd_stream(cmd, cwd):
        out = [chunk]
    return out[0] if out else ""


def _run_cmd_stream(
    cmd: list[str],
    cwd: Path | None = None,
    process_key: str | None = None,
) -> Generator[str, None, None]:
    """Запускает команду и по мере появления вывода выдаёт накопленный лог (realtime).
    Если задан process_key, процесс сохраняется в _current_processes и его можно остановить кнопкой."""
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
    if process_key:
        _current_processes[process_key] = p
    try:
        assert p.stdout is not None
        yield "Запуск: " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else "") + "\n\n"
        for line in iter(p.stdout.readline, ""):
            lines.append(line.rstrip())
            yield "\n".join(lines)
        p.wait()
        lines.append(f"\n[Статус: завершено, exit code {p.returncode}]")
        yield "\n".join(lines)
    finally:
        if process_key and _current_processes.get(process_key) is p:
            _current_processes.pop(process_key, None)


def _stop_process(process_key: str) -> None:
    """Останавливает процесс, запущенный на вкладке process_key (если есть)."""
    p = _current_processes.pop(process_key, None)
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def run_preprocess_ui(
    video_path: str | None,
    refer_path: str | None,
    folder_name: str,
    resolution: str,
    fps: int,
    replace: bool,
    wan22_dir: str,
    ckpt_dir: str,
    use_custom_preprocess: bool,
    points_file: str | None = None,
) -> Generator[str, None, None]:
    video_path = _file_path(video_path)
    refer_path = _file_path(refer_path)
    if not video_path or not Path(video_path).exists():
        yield "Укажите существующий файл видео."
        return
    if not refer_path or not Path(refer_path).exists():
        yield "Укажите существующий референс (изображение)."
        return
    name = (folder_name or "").strip() or "default"
    save_path = str(PREPROCESS_BASE / name)
    PREPROCESS_BASE.mkdir(parents=True, exist_ok=True)
    resolution = resolution.strip() or "1280 720"
    if use_custom_preprocess:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_preprocessing_custom.py"),
            "--video_path", video_path,
            "--refer_path", refer_path,
            "--save_path", save_path,
            "--resolution", resolution,
            "--fps", str(fps),
        ]
        if replace:
            cmd.append("--replace")
        if points_file and str(points_file).strip() and Path(points_file).exists():
            cmd.extend(["--points_file", str(points_file).strip()])
    else:
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
    yield from _run_cmd_stream(cmd, process_key="pre")


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
    if not src:
        yield "Выберите папку препроцессинга из списка (или укажите путь)."
        return
    if not Path(src).exists():
        yield f"Папка не найдена: {src}"
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
    yield from _run_cmd_stream(cmd, process_key="gen")


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
    yield from _run_cmd_stream(cmd, process_key="post")


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
    yield from _run_cmd_stream(cmd, process_key="up")


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
    use_relighting_lora: bool,
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
    if use_relighting_lora:
        cmd.append("--use_relighting_lora")
    yield from _run_cmd_stream(cmd, process_key="ref")


def run_thin_ui(input_dir: str, every_nth: int, out_fps: float) -> Generator[str, None, None]:
    """Прореживание кадров (30→15 fps) для генерации."""
    path = (input_dir or "").strip()
    if not path or not Path(path).exists():
        yield "Выберите папку препроцессинга (30 fps) из списка."
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_thin_for_generation.py"),
        "--input", path,
        "--every_nth", str(int(every_nth) if every_nth else 2),
        "--out_fps", str(float(out_fps) if out_fps else 15),
    ]
    yield from _run_cmd_stream(cmd, process_key="thin")


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
                        server_dir_pre = gr.Textbox(label="Папка на сервере (видео/референсы)", value="inputs")
                        with gr.Row():
                            video_server_pre = gr.Dropdown(choices=[], label="Видео с сервера", allow_custom_value=False)
                            refer_server_pre = gr.Dropdown(choices=[], label="Референс с сервера", allow_custom_value=False)
                        btn_refresh_pre = gr.Button("Обновить список с сервера")
                        video_in = gr.File(label="Или загрузить видео (поза/движение)", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        refer_in = gr.File(label="Или загрузить референс (фото)", file_types=["image"])
                        preprocess_folder = gr.Dropdown(
                            choices=_list_preprocess_folder_names(),
                            label="Имя папки (результат: preprocess/<имя>)",
                            value="default",
                            allow_custom_value=True,
                        )
                        btn_refresh_pre_folders = gr.Button("Обновить список папок")
                        preprocess_type = gr.Radio(
                            choices=["Wan2.2", "Кастомный (DWPose, как wanvideo_WanAnimate)"],
                            value="Wan2.2",
                            label="Тип препроцесса",
                        )
                        sam2_points_file_pre = gr.Dropdown(
                            choices=_list_sam2_points_files(),
                            label="Файл точек SAM2 (для кастомного replace, вкладка «Точки для SAM2»)",
                            allow_custom_value=True,
                        )
                        btn_refresh_sam2_points = gr.Button("Обновить список файлов точек")
                        resolution = gr.Textbox(label="Разрешение (W H)", value="1280 720")
                        fps = gr.Number(label="FPS (30 — полная поза DWPose, затем Thin→Generation→RIFE)", value=30, precision=0)
                        replace_pre = gr.Checkbox(label="Режим Replace (замена актёра)", value=False)
                        wan22_dir = gr.Textbox(label="Wan2.2 (папка)", value="Wan2.2")
                        ckpt_dir = gr.Textbox(label="Чекпоинт (папка)", value="Wan2.2-Animate-14B")
                        with gr.Row():
                            btn_pre = gr.Button("Запустить Preprocess")
                            btn_stop_pre = gr.Button("Остановить")
                    with gr.Column(scale=1):
                        log_pre = gr.Textbox(label="Лог", lines=20, max_lines=40)
                def refresh_pre(sd):
                    v, r = _refresh_server_inputs(sd)
                    return gr.update(choices=v), gr.update(choices=r)
                def refresh_pre_folders():
                    return gr.update(choices=_list_preprocess_folder_names())
                btn_refresh_pre.click(fn=refresh_pre, inputs=[server_dir_pre], outputs=[video_server_pre, refer_server_pre])
                btn_refresh_pre_folders.click(fn=refresh_pre_folders, outputs=[preprocess_folder])
                def refresh_sam2_points_pre():
                    return gr.update(choices=_list_sam2_points_files())
                btn_refresh_sam2_points.click(fn=refresh_sam2_points_pre, outputs=[sam2_points_file_pre])
                def do_preprocess(vs, rs, v, r, folder, res, f, rep, w, c, use_custom, points_f):
                    yield from run_preprocess_ui(
                        _resolve_input(vs, v), _resolve_input(rs, r), folder, res, int(f) if f is not None else 30, rep, w, c,
                        use_custom_preprocess=(use_custom == "Кастомный (DWPose, как wanvideo_WanAnimate)"),
                        points_file=points_f or None,
                    )
                btn_pre.click(
                    fn=do_preprocess,
                    inputs=[video_server_pre, refer_server_pre, video_in, refer_in, preprocess_folder, resolution, fps, replace_pre, wan22_dir, ckpt_dir, preprocess_type, sam2_points_file_pre],
                    outputs=[log_pre],
                )
                btn_stop_pre.click(fn=lambda: _stop_process("pre"), inputs=[], outputs=[])

            # --- Точки для SAM2 (редактор точек для маски в кастомном препроцессе) ---
            with gr.Tab("Точки для SAM2"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Клик по изображению** добавляет координаты в поле ниже → нажмите «Добавить точку». Зелёный = объект, красный = фон. Сохраните и укажите файл в кастомном препроцессе (Replace).")
                        sam2_video = gr.Dropdown(
                            choices=_refresh_server_inputs("inputs")[0],
                            label="Видео (для первого кадра)",
                            allow_custom_value=True,
                        )
                        sam2_video_file = gr.File(label="Или загрузить видео", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        btn_sam2_load_frame = gr.Button("Загрузить первый кадр")
                        sam2_canvas = gr.HTML(value="<p style='color:#666'>Загрузите первый кадр.</p>")
                        sam2_frame_info = gr.Textbox(label="Инфо", interactive=False)
                        sam2_click_xy = gr.Textbox(
                            label="Клик на canvas (x, y) — кликните по изображению выше",
                            value="",
                            interactive=False,
                            elem_id="sam2_click_xy",
                        )
                        sam2_point_type = gr.Radio(choices=["Положительная", "Отрицательная"], value="Положительная", label="Тип точки")
                        with gr.Row():
                            btn_sam2_add = gr.Button("Добавить точку")
                            btn_sam2_del = gr.Button("Удалить последнюю")
                        sam2_points_state = gr.State(value=[])
                        sam2_base_image = gr.State(value=None)
                        sam2_points_text = gr.Textbox(label="Точки (x,y,1=полож/0=отриц)", lines=6, max_lines=12)
                        sam2_points_name = gr.Textbox(label="Имя файла точек (сохранится preprocess/sam2_points_<имя>.json)", value="my_run")
                        btn_sam2_save = gr.Button("Сохранить точки в файл")
                        sam2_save_status = gr.Textbox(label="Статус", interactive=False)
                    with gr.Column(scale=0):
                        pass
                def _points_to_text(pts):
                    if not pts:
                        return ""
                    return "\n".join(f"{p['x']},{p['y']},{1 if p['positive'] else 0}" for p in pts)
                def _draw_points_on_image(img, pts):
                    if img is None or not pts:
                        return img
                    try:
                        import cv2
                        out = img.copy()
                        for p in pts:
                            x, y = int(p["x"]), int(p["y"])
                            color = (0, 255, 0) if p["positive"] else (255, 0, 0)
                            cv2.circle(out, (x, y), 8, color, 2)
                        return out
                    except Exception:
                        return img
                def _image_to_clickable_html(img, points):
                    """Рисует точки на img и возвращает HTML с canvas: клик записывает x,y в #sam2_click_xy input."""
                    import base64
                    import io
                    if img is None:
                        return "<p style='color:#666'>Нет изображения.</p>"
                    drawn = _draw_points_on_image(img, points)
                    if drawn is None:
                        drawn = img
                    is_rgb = drawn.ndim == 3 and drawn.shape[-1] == 3
                    if not is_rgb:
                        drawn = np.stack([drawn] * 3, axis=-1) if drawn.ndim == 2 else drawn
                    pil_img = None
                    try:
                        from PIL import Image
                        arr = (np.clip(drawn, 0, 255) if drawn.dtype != np.uint8 else drawn).astype(np.uint8)
                        pil_img = Image.fromarray(arr)
                    except Exception:
                        return "<p style='color:#666'>Ошибка конвертации.</p>"
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return (
                        f'<div style="margin:0.5em 0"><img id="sam2_canvas_img" src="data:image/png;base64,{b64}" '
                        'style="max-width:100%; cursor:crosshair; display:block" '
                        'title="Кликните, чтобы задать координаты точки"/>'
                        '<script>'
                        "(function(){ var el = document.getElementById('sam2_canvas_img'); if(!el) return; "
                        "el.onclick = function(e){ var r = el.getBoundingClientRect(); var x = Math.round(e.clientX - r.left); var y = Math.round(e.clientY - r.top); "
                        "var inp = document.querySelector('#sam2_click_xy input, [id*=\\'sam2_click_xy\\'] input'); "
                        "if(inp){ inp.value = x + ',' + y; inp.dispatchEvent(new Event('input', {bubbles: true})); } "
                        "}); })();"
                        "</script></div>"
                    )
                def load_first_frame(v_path, f_path):
                    path = _resolve_input(v_path, f_path)
                    img, msg = _extract_first_frame(path)
                    if img is None:
                        return "<p style='color:#888'>Укажите видео и нажмите «Загрузить первый кадр».</p>", msg, [], None
                    html = _image_to_clickable_html(img, [])
                    return html, msg, [], img
                def add_point(base_img, state, click_xy, ptype):
                    x, y = None, None
                    if click_xy and isinstance(click_xy, str):
                        parts = str(click_xy).strip().replace(" ", "").split(",")
                        if len(parts) >= 2:
                            try:
                                x, y = int(round(float(parts[0]))), int(round(float(parts[1])))
                            except ValueError:
                                pass
                    if x is None or y is None:
                        return _image_to_clickable_html(base_img, state) if base_img is not None else "", state, _points_to_text(state), ""
                    state = list(state) + [{"x": x, "y": y, "positive": ptype == "Положительная"}]
                    html = _image_to_clickable_html(base_img, state) if base_img is not None else ""
                    return html, state, _points_to_text(state), ""
                def del_last_point(base_img, state):
                    if not state:
                        return (_image_to_clickable_html(base_img, state) if base_img is not None else ""), state, _points_to_text(state)
                    state = list(state)[:-1]
                    html = _image_to_clickable_html(base_img, state) if base_img is not None else ""
                    return html, state, _points_to_text(state)
                def save_points(state, name, base_img):
                    if not name or not name.strip():
                        return "Укажите имя файла"
                    name = name.strip().replace(" ", "_")
                    if not name:
                        return "Укажите имя"
                    PREPROCESS_BASE.mkdir(parents=True, exist_ok=True)
                    path = PREPROCESS_BASE / f"sam2_points_{name}.json"
                    try:
                        data = {"points": state}
                        if base_img is not None and hasattr(base_img, "shape") and len(base_img.shape) >= 2:
                            data["image_width"] = int(base_img.shape[1])
                            data["image_height"] = int(base_img.shape[0])
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        return f"Сохранено: {path}"
                    except Exception as e:
                        return str(e)
                btn_sam2_load_frame.click(
                    fn=load_first_frame,
                    inputs=[sam2_video, sam2_video_file],
                    outputs=[sam2_canvas, sam2_frame_info, sam2_points_state, sam2_base_image],
                )
                btn_sam2_add.click(
                    fn=add_point,
                    inputs=[sam2_base_image, sam2_points_state, sam2_click_xy, sam2_point_type],
                    outputs=[sam2_canvas, sam2_points_state, sam2_points_text, sam2_click_xy],
                )
                btn_sam2_del.click(
                    fn=del_last_point,
                    inputs=[sam2_base_image, sam2_points_state],
                    outputs=[sam2_canvas, sam2_points_state, sam2_points_text],
                )
                btn_sam2_save.click(
                    fn=save_points,
                    inputs=[sam2_points_state, sam2_points_name, sam2_base_image],
                    outputs=[sam2_save_status],
                )
                btn_sam2_refresh_videos = gr.Button("Обновить список видео")
                btn_sam2_refresh_videos.click(fn=lambda: gr.update(choices=_refresh_server_inputs("inputs")[0]), outputs=[sam2_video])

            # --- Thin (30→15 fps для генерации) ---
            with gr.Tab("1.5 Thin"):
                with gr.Row():
                    with gr.Column(scale=1):
                        thin_input = gr.Dropdown(
                            choices=_list_preprocess_folders_for_thin(),
                            label="Папка препроцессинга (30 fps)",
                            allow_custom_value=True,
                        )
                        btn_refresh_thin = gr.Button("Обновить список")
                        thin_every_nth = gr.Number(label="Брать каждый n-й кадр", value=2, precision=0)
                        thin_out_fps = gr.Number(label="Выходной FPS", value=15, precision=0)
                        with gr.Row():
                            btn_thin = gr.Button("Проредить (30→15 fps)")
                            btn_stop_thin = gr.Button("Остановить")
                    with gr.Column(scale=1):
                        log_thin = gr.Textbox(label="Лог", lines=20, max_lines=40)
                gr.Markdown("Создаётся папка `<выбранная>_gen` для этапа Generation. Требуется ffmpeg.")
                def refresh_thin():
                    return gr.update(choices=_list_preprocess_folders_for_thin())
                btn_refresh_thin.click(fn=refresh_thin, outputs=[thin_input])
                def do_thin(inp, nth, fps_val):
                    yield from run_thin_ui(inp, nth, fps_val)
                btn_thin.click(
                    fn=do_thin,
                    inputs=[thin_input, thin_every_nth, thin_out_fps],
                    outputs=[log_thin],
                )
                btn_stop_thin.click(fn=lambda: _stop_process("thin"), inputs=[], outputs=[])

            # --- Generation ---
            with gr.Tab("2. Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        src_root = gr.Dropdown(
                            choices=_list_preprocess_folders(),
                            label="Папка препроцессинга (src_root_path)",
                            allow_custom_value=True,
                        )
                        btn_refresh_src = gr.Button("Обновить список папок")
                        save_file = gr.Textbox(label="Имя выходного файла (сохраняется в out/)", value="output.mp4")
                        prompt = gr.Textbox(label="Промпт (пусто = дефолт Wan2.2)")
                        sample_steps = gr.Number(label="Steps", value=30, precision=0)
                        sample_guide_scale = gr.Number(label="CFG", value=2.5)
                        sample_shift = gr.Number(label="Shift", value=5)
                        no_offload = gr.Checkbox(label="Не выгружать модель (--no_offload, быстрее)", value=False)
                        replace_gen = gr.Checkbox(label="Replace", value=False)
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        wan22_dir_gen = gr.Textbox(label="Wan2.2", value="Wan2.2")
                        ckpt_dir_gen = gr.Textbox(label="Чекпоинт", value="Wan2.2-Animate-14B")
                        with gr.Row():
                            btn_gen = gr.Button("Запустить Generation")
                            btn_stop_gen = gr.Button("Остановить")
                    with gr.Column(scale=1):
                        log_gen = gr.Textbox(label="Лог", lines=20, max_lines=40)
                def refresh_src_folders():
                    return gr.update(choices=_list_preprocess_folders())
                btn_refresh_src.click(fn=refresh_src_folders, outputs=[src_root])
                def do_generation(src, sf, p, st, cfg, sh, no, rep, se, w, c):
                    yield from run_generation_ui(
                        src, sf, p or None,
                        int(st) if st is not None else None,
                        float(cfg) if cfg is not None else None,
                        float(sh) if sh is not None else None,
                        no, rep, int(se) if se is not None else 42, w, c,
                    )
                btn_gen.click(
                    fn=do_generation,
                    inputs=[src_root, save_file, prompt, sample_steps, sample_guide_scale, sample_shift,
                            no_offload, replace_gen, seed, wan22_dir_gen, ckpt_dir_gen],
                    outputs=[log_gen],
                )
                btn_stop_gen.click(fn=lambda: _stop_process("gen"), inputs=[], outputs=[])

            # --- Refine (второй replace для артефактов) ---
            with gr.Tab("Refine"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_server_dir = gr.Textbox(label="Папка на сервере", value="inputs")
                        ref_video_server = gr.Dropdown(choices=[], label="Видео с сервера (или загрузите)", allow_custom_value=False)
                        ref_refer_server = gr.Dropdown(choices=[], label="Референс с сервера (или загрузите)", allow_custom_value=False)
                        btn_refresh_ref = gr.Button("Обновить список с сервера")
                        ref_in = gr.File(label="Или загрузить видео для рефайна", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        ref_refer = gr.File(label="Или загрузить референс", file_types=["image"])
                        ref_save_path = gr.Textbox(label="Папка препроцессинга refine", value="refine_process")
                        ref_save_file = gr.Textbox(label="Имя выходного файла (сохраняется в out/)", value="refined.mp4")
                        ref_resolution = gr.Textbox(label="Разрешение (W H)", value="1280 720")
                        ref_fps = gr.Number(label="FPS", value=16, precision=0)
                        ref_prompt = gr.Textbox(label="Промпт (пусто = дефолт)")
                        ref_steps = gr.Number(label="Steps", value=30, precision=0)
                        ref_cfg = gr.Number(label="CFG", value=2.5)
                        ref_shift = gr.Number(label="Shift", value=5)
                        ref_no_offload = gr.Checkbox(label="Не выгружать модель", value=False)
                        ref_use_relighting_lora = gr.Checkbox(label="Use relighting LoRA", value=True)
                        ref_seed = gr.Number(label="Seed", value=42, precision=0)
                        ref_wan22 = gr.Textbox(label="Wan2.2", value="Wan2.2")
                        ref_ckpt = gr.Textbox(label="Чекпоинт", value="Wan2.2-Animate-14B")
                        with gr.Row():
                            btn_ref = gr.Button("Запустить Refine")
                            btn_stop_ref = gr.Button("Остановить")
                    with gr.Column(scale=1):
                        log_ref = gr.Textbox(label="Лог", lines=20, max_lines=40)
                def refresh_ref(sd):
                    v, r = _refresh_server_inputs(sd)
                    return gr.update(choices=v), gr.update(choices=r)
                btn_refresh_ref.click(fn=refresh_ref, inputs=[ref_server_dir], outputs=[ref_video_server, ref_refer_server])
                def do_refine(ivs, irs, i, r, sp, sf, res, f, p, st, cfg, sh, no, relight, se, w, c):
                    yield from run_refine_ui(
                        _resolve_input(ivs, i), _resolve_input(irs, r), sp, sf, res, f, p or None,
                        int(st) if st is not None else None, float(cfg) if cfg is not None else None, float(sh) if sh is not None else None,
                        no, relight, se, w, c,
                    )
                btn_ref.click(
                    fn=do_refine,
                    inputs=[ref_video_server, ref_refer_server, ref_in, ref_refer, ref_save_path, ref_save_file, ref_resolution, ref_fps,
                            ref_prompt, ref_steps, ref_cfg, ref_shift, ref_no_offload, ref_use_relighting_lora, ref_seed, ref_wan22, ref_ckpt],
                    outputs=[log_ref],
                )
                btn_stop_ref.click(fn=lambda: _stop_process("ref"), inputs=[], outputs=[])

            # --- Postprocess (RIFE / ffmpeg) ---
            with gr.Tab("3. Postprocess (FPS)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        post_in_server = gr.Dropdown(choices=_list_output_files(), label="Входное видео с сервера (Wan2.2/out, out/)", allow_custom_value=False)
                        post_in = gr.File(label="Или загрузить видео", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        post_out = gr.Textbox(label="Выходное видео (относительный путь → out/)", value="out_30fps.mp4")
                        target_fps = gr.Number(label="Целевой FPS", value=30, precision=0)
                        backend_post = gr.Radio(label="Бэкенд", choices=["vulkan", "cuda", "ffmpeg"], value="vulkan")
                        no_audio_post = gr.Checkbox(label="Без аудио", value=False)
                        with gr.Row():
                            btn_post = gr.Button("Запустить Postprocess")
                            btn_stop_post = gr.Button("Остановить")
                    with gr.Column(scale=1):
                        log_post = gr.Textbox(label="Лог", lines=20, max_lines=40)
                def do_postprocess(si, i, o, f, b, na):
                    yield from run_postprocess_ui(
                        _resolve_input(si, i), o, int(f) if f is not None else 30, b, na,
                    )
                btn_post.click(
                    fn=do_postprocess,
                    inputs=[post_in_server, post_in, post_out, target_fps, backend_post, no_audio_post],
                    outputs=[log_post],
                )
                btn_stop_post.click(fn=lambda: _stop_process("post"), inputs=[], outputs=[])

            # --- Upscale ---
            with gr.Tab("4. Upscale"):
                with gr.Row():
                    with gr.Column(scale=1):
                        up_in_server = gr.Dropdown(choices=_list_output_files(), label="Входное видео с сервера", allow_custom_value=False)
                        up_in = gr.File(label="Или загрузить видео", file_types=[".mp4", ".mov", ".avi", ".webm"])
                        up_out = gr.Textbox(label="Выходное видео (относительный путь → out/)", value="upscaled.mp4")
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
                        with gr.Row():
                            btn_up = gr.Button("Запустить Upscale")
                            btn_stop_up = gr.Button("Остановить")
                    with gr.Column(scale=1):
                        log_up = gr.Textbox(label="Лог", lines=20, max_lines=40)
                def do_upscale(si, i, o, b, rh, rw, sc, mn, na):
                    yield from run_upscale_ui(
                        _resolve_input(si, i), o, b,
                        int(rh) if rh is not None else 720,
                        int(rw) if rw is not None else 1280,
                        float(sc) if sc is not None else 2,
                        mn or "realesr-general-x4v3",
                        na,
                    )
                btn_up.click(
                    fn=do_upscale,
                    inputs=[up_in_server, up_in, up_out, backend_up, res_h, res_w, outscale, model_name, no_audio_up],
                    outputs=[log_up],
                )
                btn_stop_up.click(fn=lambda: _stop_process("up"), inputs=[], outputs=[])

            # --- Скачать результаты ---
            with gr.Tab("Скачать"):
                with gr.Row():
                    with gr.Column(scale=1):
                        download_dropdown = gr.Dropdown(
                            choices=_list_output_files(),
                            label="Готовые результаты (Wan2.2/out, out/)",
                            allow_custom_value=False,
                        )
                        btn_download_refresh = gr.Button("Обновить список")
                        download_out = gr.File(label="Скачать файл", interactive=False)
                def refresh_downloads():
                    return gr.update(choices=_list_output_files())
                def on_select_download(path):
                    if path and Path(path).exists():
                        return path
                    return None
                btn_download_refresh.click(fn=refresh_downloads, outputs=[download_dropdown])
                download_dropdown.change(fn=on_select_download, inputs=[download_dropdown], outputs=[download_out])

    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
