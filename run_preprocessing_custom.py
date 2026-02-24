#!/usr/bin/env python3
"""
Кастомный препроцесс в стиле wanvideo_WanAnimate (ComfyUI): DWPose + маска (GrowMask, BlockifyMask).

Не использует Wan2.2 preprocess_data.py. Выход в формате Wan2.2: src_pose.mp4, src_face.mp4,
src_ref.png; для replace: src_bg.mp4, src_mask.mp4 (маска: из позы body bbox → grow 10px → blockify 32px).

Требуется: pip install dwposes opencv-python imageio imageio-ffmpeg
Опционально: точки для SAM2 из файла (пока маска строится по bbox тела из DWPose).

Использование:
  python run_preprocessing_custom.py --video_path v.mp4 --refer_path r.jpg --save_path preprocess/my_run
  python run_preprocessing_custom.py --video_path v.mp4 --refer_path r.jpg --save_path preprocess/my_run --replace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess import (
    load_image,
    load_video_frames,
    resize_lanczos,
    save_image,
)


def _get_dwpose_detector(det_path: str | None = None, pose_path: str | None = None):
    try:
        from DWPoses import DWposeDetector
    except ImportError:
        raise RuntimeError(
            "Установите DWPose: pip install dwposes\n"
            "Кастомный препроцесс использует DWPose (как в wanvideo_WanAnimate), не Wan2.2."
        )
    if det_path and pose_path:
        return DWposeDetector(det=det_path, pose=pose_path)
    return DWposeDetector()


def _frames_to_uint8_rgb(frames: np.ndarray) -> list[np.ndarray]:
    """(T,H,W,C) [0,1] -> list of (H,W,3) uint8 RGB."""
    out = []
    for t in range(frames.shape[0]):
        arr = np.clip(frames[t], 0, 1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        out.append((arr * 255).astype(np.uint8))
    return out


def _grow_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Расширение маски на pixels (как GrowMask)."""
    if pixels <= 0:
        return mask
    try:
        import cv2
    except ImportError:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    m = (mask > 0.5).astype(np.uint8)
    m = cv2.dilate(m, kernel)
    return m.astype(np.float32)


def _blockify_mask(mask: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Выравнивание маски по блокам block_size (как BlockifyMask)."""
    h, w = mask.shape[:2]
    if block_size <= 0:
        return mask
    bh, bw = (h + block_size - 1) // block_size, (w + block_size - 1) // block_size
    out = np.zeros((bh * block_size, bw * block_size), dtype=np.float32)
    out[:h, :w] = mask
    blocks = out.reshape(bh, block_size, bw, block_size)
    block_mean = blocks.mean(axis=(1, 3))
    for i in range(bh):
        for j in range(bw):
            out[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = (
                1.0 if block_mean[i, j] > 0.5 else 0.0
            )
    return out[:h, :w]


def _body_bbox_from_pose(pose: dict, height: int, width: int, margin: int = 20) -> tuple[int, int, int, int]:
    """Bbox по ключевым точкам тела из DWPose (bodies): dict с candidate или numpy (K,2)/(K,3)."""
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    if "bodies" in pose and pose["bodies"]:
        for body in pose["bodies"]:
            if hasattr(body, "shape"):
                pts = np.asarray(body)
                if pts.ndim == 2 and pts.size >= 2:
                    xs = pts[:, 0]
                    ys = pts[:, 1]
                    if pts.shape[1] >= 3:
                        conf = pts[:, 2]
                        xs = xs[conf > 0.2]
                        ys = ys[conf > 0.2]
                    if len(xs) > 0:
                        x_min = min(x_min, float(xs.min()))
                        y_min = min(y_min, float(ys.min()))
                        x_max = max(x_max, float(xs.max()))
                        y_max = max(y_max, float(ys.max()))
                continue
            if isinstance(body, dict) and "candidate" in body:
                pts = body["candidate"]
            elif isinstance(body, (list, tuple)):
                pts = body
            else:
                continue
            for pt in pts:
                if len(pt) >= 2 and float(pt[2] if len(pt) >= 3 else 1) > 0.2:
                    x, y = float(pt[0]), float(pt[1])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
    if x_max <= x_min or y_max <= y_min:
        return 0, 0, width, height
    x_min = max(0, int(x_min) - margin)
    y_min = max(0, int(y_min) - margin)
    x_max = min(width, int(x_max) + margin)
    y_max = min(height, int(y_max) + margin)
    return x_min, y_min, x_max, y_max


def _mask_from_bbox(height: int, width: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    m = np.zeros((height, width), dtype=np.float32)
    m[y_min:y_max, x_min:x_max] = 1.0
    return m


def _load_points_file(path: Path) -> tuple[list[dict], int, int]:
    """Загружает точки из JSON. Возвращает (points, image_width, image_height)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        points = data
        img_w, img_h = 1280, 720
    else:
        points = data.get("points", data)
        img_w = int(data.get("image_width", 1280))
        img_h = int(data.get("image_height", 720))
    return points, img_w, img_h


def _mask_from_points(
    height: int,
    width: int,
    points: list[dict],
    src_width: int,
    src_height: int,
    point_radius: int = 40,
) -> np.ndarray:
    """Маска по точкам: положительные — круги в объединении, отрицательные — вычитаем. Точки в координатах src."""
    try:
        import cv2
    except ImportError:
        return np.zeros((height, width), dtype=np.float32)
    scale_x = width / max(1, src_width)
    scale_y = height / max(1, src_height)
    m = np.zeros((height, width), dtype=np.float32)
    pos_pts = [(int(p["x"] * scale_x), int(p["y"] * scale_y)) for p in points if p.get("positive", True)]
    neg_pts = [(int(p["x"] * scale_x), int(p["y"] * scale_y)) for p in points if not p.get("positive", True)]
    r = max(5, int(point_radius * min(scale_x, scale_y)))
    for (x, y) in pos_pts:
        x, y = max(0, min(width, x)), max(0, min(height, y))
        cv2.circle(m, (x, y), r, 1.0, -1)
    for (x, y) in neg_pts:
        x, y = max(0, min(width, x)), max(0, min(height, y))
        cv2.circle(m, (x, y), r, 0.0, -1)
    return np.clip(m, 0, 1).astype(np.float32)


def _face_bbox_from_pose(pose: dict, height: int, width: int, scale: float = 1.5) -> tuple[int, int, int, int]:
    """Bbox лица из DWPose (faces: numpy (68,2) или list точек; иначе body keypoints 0-10)."""
    pts = []
    if "faces" in pose and pose["faces"]:
        for face in pose["faces"]:
            if hasattr(face, "shape"):
                arr = np.asarray(face)
                if arr.ndim == 2 and arr.size >= 2:
                    for i in range(arr.shape[0]):
                        pts.append((float(arr[i, 0]), float(arr[i, 1])))
                continue
            if isinstance(face, (list, tuple)):
                for pt in face:
                    if len(pt) >= 2:
                        pts.append((float(pt[0]), float(pt[1])))
            elif isinstance(face, dict):
                for k, v in face.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        pts.append((float(v[0]), float(v[1])))
    if "bodies" in pose and pose["bodies"] and len(pts) == 0:
        body = pose["bodies"][0]
        if hasattr(body, "shape"):
            arr = np.asarray(body)
            if arr.ndim == 2 and arr.shape[0] >= 11:
                for i in range(min(11, arr.shape[0])):
                    if arr.shape[1] >= 3 and float(arr[i, 2]) > 0.2:
                        pts.append((float(arr[i, 0]), float(arr[i, 1])))
                    elif arr.shape[1] == 2:
                        pts.append((float(arr[i, 0]), float(arr[i, 1])))
        elif isinstance(body, (list, tuple)):
            for i, pt in enumerate(body[:11]):
                if len(pt) >= 2 and float(pt[2] if len(pt) >= 3 else 1) > 0.2:
                    pts.append((float(pt[0]), float(pt[1])))
    if not pts:
        cx, cy = width // 2, height // 4
        side = min(width, height) // 4
        return max(0, cx - side), max(0, cy - side), min(width, cx + side), min(height, cy + side)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    r = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * scale
    r = min(r, min(width, height) / 2)
    x_min = max(0, int(cx - r))
    y_min = max(0, int(cy - r))
    x_max = min(width, int(cx + r))
    y_max = min(height, int(cy + r))
    return x_min, y_min, x_max, y_max


def run_custom_preprocess(
    video_path: Path,
    refer_path: Path,
    save_path: Path,
    resolution: tuple[int, int] = (1280, 720),
    fps: int = 30,
    replace: bool = False,
    grow_mask_px: int = 10,
    blockify_size: int = 32,
    det_onnx: str | None = None,
    pose_onnx: str | None = None,
    points_file: Path | str | None = None,
) -> int:
    """
    Кастомный препроцесс: DWPose → pose/face видео, ref; при replace — маска (bbox или из точек→grow→blockify).
    Если задан points_file (JSON из UI «Точки для SAM2»), маска строится по точкам.
    Возвращает 0 при успехе.
    """
    save_path = Path(save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    w, h = resolution[0], resolution[1]

    print("Загрузка видео...")
    frames = load_video_frames(video_path, target_size=(w, h))
    n_frames = frames.shape[0]
    print(f"Кадров: {n_frames}, разрешение {w}x{h}")

    print("Инициализация DWPose...")
    detector = _get_dwpose_detector(det_onnx, pose_onnx)

    frames_uint8 = _frames_to_uint8_rgb(frames)
    pose_images = []
    face_crops = []
    body_masks = []

    points_list = []
    src_w, src_h = w, h
    if points_file and Path(points_file).exists():
        points_list, src_w, src_h = _load_points_file(Path(points_file))
        print(f"Маска по точкам из {points_file} (точек: {len(points_list)}, размер кадра в UI: {src_w}x{src_h})")

    for t in range(n_frames):
        if (t + 1) % 20 == 0 or t == 0:
            print(f"  DWPose кадр {t + 1}/{n_frames}")
        img = frames_uint8[t]
        try:
            pose, imgout = detector.predict(img)
        except Exception as e:
            print(f"  Предупреждение: кадр {t}: {e}, используем исходный кадр")
            imgout = img
            pose = {}
        if isinstance(imgout, np.ndarray):
            if imgout.ndim == 2:
                imgout = np.stack([imgout] * 3, axis=-1)
            if imgout.shape[-1] == 4:
                imgout = imgout[..., :3]
            try:
                import cv2
                if imgout.shape[-1] == 3:
                    imgout = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
            except ImportError:
                pass
            pose_img = np.array(imgout, dtype=np.float32) / 255.0
        else:
            pose_img = np.array(imgout.convert("RGB"), dtype=np.float32) / 255.0
        pose_images.append(pose_img)

        face_bbox = _face_bbox_from_pose(pose, h, w)
        x1, y1, x2, y2 = face_bbox
        face_crop = frames[t, y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = frames[t]
        if face_crop.shape[0] != 512 or face_crop.shape[1] != 512:
            face_crop = resize_lanczos(
                face_crop if face_crop.ndim == 3 else np.stack([face_crop] * 3, axis=-1),
                512, 512,
            )
        face_crops.append(face_crop)

        if replace:
            if points_list:
                m = _mask_from_points(h, w, points_list, src_w, src_h)
            else:
                bbox = _body_bbox_from_pose(pose, h, w)
                m = _mask_from_bbox(h, w, bbox)
            m = _grow_mask(m, grow_mask_px)
            m = _blockify_mask(m, blockify_size)
            body_masks.append(m)

    pose_stack = np.stack(pose_images, axis=0)
    face_stack = np.stack(face_crops, axis=0)

    try:
        import imageio
    except ImportError:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_float = float(fps)
        out_pose = cv2.VideoWriter(
            str(save_path / "src_pose.mp4"), fourcc, fps_float, (w, h)
        )
        for t in range(n_frames):
            fr = (np.clip(pose_stack[t], 0, 1) * 255).astype(np.uint8)
            out_pose.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        out_pose.release()
        out_face = cv2.VideoWriter(
            str(save_path / "src_face.mp4"), fourcc, fps_float, (512, 512)
        )
        for t in range(n_frames):
            fr = (np.clip(face_stack[t], 0, 1) * 255).astype(np.uint8)
            out_face.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        out_face.release()
    else:
        writer_pose = imageio.get_writer(save_path / "src_pose.mp4", fps=fps, codec="libx264", quality=8)
        for t in range(n_frames):
            writer_pose.append_data((np.clip(pose_stack[t], 0, 1) * 255).astype(np.uint8))
        writer_pose.close()
        writer_face = imageio.get_writer(save_path / "src_face.mp4", fps=fps, codec="libx264", quality=8)
        for t in range(n_frames):
            writer_face.append_data((np.clip(face_stack[t], 0, 1) * 255).astype(np.uint8))
        writer_face.close()

    ref_img = load_image(refer_path)
    ref_img = resize_lanczos(ref_img, w, h)
    save_image(ref_img, save_path / "src_ref.png")
    print("Сохранено: src_pose.mp4, src_face.mp4, src_ref.png")

    if replace and body_masks:
        mask_stack = np.stack(body_masks, axis=0)
        bg_frames = []
        mask_vis_frames = []
        for t in range(n_frames):
            m = mask_stack[t]
            bg = frames[t] * (1 - m[:, :, np.newaxis])
            bg_frames.append(bg)
            mask_vis = np.stack([m, m, m], axis=-1)
            mask_vis_frames.append(mask_vis)
        bg_stack = np.stack(bg_frames, axis=0)
        mask_vis_stack = np.stack(mask_vis_frames, axis=0)
        try:
            import imageio
            w_bg = imageio.get_writer(save_path / "src_bg.mp4", fps=fps, codec="libx264", quality=8)
            w_mask = imageio.get_writer(save_path / "src_mask.mp4", fps=fps, codec="libx264", quality=8)
            for t in range(n_frames):
                w_bg.append_data((np.clip(bg_stack[t], 0, 1) * 255).astype(np.uint8))
                w_mask.append_data((np.clip(mask_vis_stack[t], 0, 1) * 255).astype(np.uint8))
            w_bg.close()
            w_mask.close()
        except ImportError:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_bg = cv2.VideoWriter(str(save_path / "src_bg.mp4"), fourcc, float(fps), (w, h))
            out_mask = cv2.VideoWriter(str(save_path / "src_mask.mp4"), fourcc, float(fps), (w, h))
            for t in range(n_frames):
                out_bg.write(cv2.cvtColor((np.clip(bg_stack[t], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                out_mask.write(cv2.cvtColor((np.clip(mask_vis_stack[t], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            out_bg.release()
            out_mask.release()
        print("Сохранено: src_bg.mp4, src_mask.mp4 (replace)")

    print("Кастомный препроцесс завершён:", save_path)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Кастомный препроцесс (DWPose, как wanvideo_WanAnimate), без Wan2.2"
    )
    parser.add_argument("--video_path", type=Path, required=True, help="Входное видео")
    parser.add_argument("--refer_path", type=Path, required=True, help="Референсное изображение")
    parser.add_argument("--save_path", type=Path, default=Path("preprocess/custom"), help="Папка результата")
    parser.add_argument("--resolution", type=str, default="1280 720", help="Разрешение W H")
    parser.add_argument("--fps", type=int, default=30, help="FPS выходных видео")
    parser.add_argument("--replace", action="store_true", help="Режим replace: маска (bbox→grow→blockify)")
    parser.add_argument("--grow_mask_px", type=int, default=10, help="GrowMask: расширение маски (px)")
    parser.add_argument("--blockify_size", type=int, default=32, help="BlockifyMask: размер блока")
    parser.add_argument("--det_onnx", type=str, default=None, help="Путь к yolox_l.onnx (опционально)")
    parser.add_argument("--pose_onnx", type=str, default=None, help="Путь к dw-ll_ucoco_384.onnx (опционально)")
    parser.add_argument("--points_file", type=Path, default=None, help="JSON с точками для маски (из UI «Точки для SAM2»)")
    args = parser.parse_args()

    res = [int(x) for x in args.resolution.split()]
    resolution = (res[0], res[1]) if len(res) >= 2 else (1280, 720)

    return run_custom_preprocess(
        video_path=args.video_path,
        refer_path=args.refer_path,
        save_path=args.save_path,
        resolution=resolution,
        fps=args.fps,
        replace=args.replace,
        grow_mask_px=args.grow_mask_px,
        blockify_size=args.blockify_size,
        det_onnx=args.det_onnx,
        pose_onnx=args.pose_onnx,
        points_file=args.points_file,
    )


if __name__ == "__main__":
    sys.exit(main())
