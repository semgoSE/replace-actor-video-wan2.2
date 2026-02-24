"""
Утилиты препроцессинга (чистый Python: NumPy/PIL).
- Загрузка/сохранение изображений и видео
- Resize (Lanczos)
Используются при необходимости кастомного препроцессинга; основной пайплайн
использует официальный preprocess_data.py из Wan2.2 (этап 1).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Загрузка изображения в HWC, [0,1], RGB."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Install Pillow: pip install Pillow")
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def save_image(arr: np.ndarray, path: Union[str, Path]) -> None:
    """Сохранение HWC [0,1] в файл."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Install Pillow: pip install Pillow")
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def resize_lanczos(
    img: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Resize HWC [0,1] с Lanczos (как в workflow ImageResizeKJv2)."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Install Pillow: pip install Pillow")
    h, w = img.shape[:2]
    if w == width and h == height:
        return img.copy()
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def resize_to_fit(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    pad: bool = True,
    pad_value: float = 0.0,
    align: str = "top",
) -> np.ndarray:
    """
    Масштабирование с сохранением пропорций и опциональной подложкой.
    align: "top", "center", "bottom"
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if not PIL_AVAILABLE:
        raise RuntimeError("Install Pillow: pip install Pillow")
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = np.array(pil, dtype=np.float32) / 255.0
    if not pad or (new_w == target_w and new_h == target_h):
        return out
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.float32)
    y0 = 0
    if align == "center":
        y0 = (target_h - new_h) // 2
    elif align == "bottom":
        y0 = target_h - new_h
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = out
    return canvas


def load_video_frames(
    path: Union[str, Path],
    max_frames: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Загрузка кадров из видео. Возвращает (T, H, W, C) [0,1].
    Требует: pip install imageio imageio-ffmpeg (или opencv-python).
    """
    try:
        import imageio
    except ImportError:
        try:
            import cv2
        except ImportError:
            raise RuntimeError("Install imageio + imageio-ffmpeg or opencv-python for video load")
    try:
        reader = imageio.get_reader(path, "ffmpeg")
        frames = []
        for i, fr in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            if len(fr.shape) == 2:
                fr = np.stack([fr] * 3, axis=-1)
            elif fr.shape[-1] == 4:
                fr = fr[..., :3]
            frames.append(np.array(fr, dtype=np.float32) / 255.0)
        reader.close()
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if max_frames is not None and len(frames) >= max_frames:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(np.array(fr, dtype=np.float32) / 255.0)
        cap.release()
    if not frames:
        raise ValueError(f"No frames in video: {path}")
    out = np.stack(frames, axis=0)
    if target_size is not None:
        tw, th = target_size
        out = np.stack([resize_lanczos(out[t], tw, th) for t in range(out.shape[0])], axis=0)
    return out


def numpy_to_comfy_image(arr: np.ndarray) -> List[List[int]]:
    """
    Конвертация numpy (T,H,W,C) или (H,W,C) [0,1] в формат ComfyUI API image.
    Comfy ожидает list of list: [{"filename": "name.png", "subfolder": "", "type": "input"}]
    и отдельно загруженные файлы через /upload/image.
    Для API мы сохраняем во временный файл и возвращаем имя; здесь возвращаем только shape info.
    """
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    # (T, H, W, C)
    return [arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3]]


def prepare_reference_image(
    path: Union[str, Path],
    width: int = 832,
    height: int = 480,
) -> np.ndarray:
    """Reference image: resize к 832x480 (lanczos), как Set_reference_image + ImageResizeKJv2."""
    img = load_image(path)
    return resize_lanczos(img, width, height)


def prepare_background_image(
    path: Union[str, Path],
    width: int = 832,
    height: int = 480,
    num_frames: int = 501,
) -> np.ndarray:
    """Background: одно изображение ресайз, затем повтор на num_frames. (T,H,W,C)."""
    img = load_image(path)
    img = resize_lanczos(img, width, height)
    return np.tile(img[np.newaxis, ...], (num_frames, 1, 1, 1))


def get_pose_frames_from_video(
    video_path: Union[str, Path],
    width: int = 832,
    height: int = 480,
    num_frames: Optional[int] = None,
) -> np.ndarray:
    """Pose images из видео: загрузить кадры и ресайз к 832x480."""
    return load_video_frames(video_path, max_frames=num_frames, target_size=(width, height))


def get_pose_frames_from_dir(
    dir_path: Union[str, Path],
    width: int = 832,
    height: int = 480,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> np.ndarray:
    """Pose images из папки с кадрами (сортируем по имени)."""
    dir_path = Path(dir_path)
    files = sorted(
        [f for f in dir_path.iterdir() if f.suffix.lower() in extensions],
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"No images in {dir_path}")
    frames = [resize_lanczos(load_image(f), width, height) for f in files]
    return np.stack(frames, axis=0)


def crop_by_mask_center(
    img: np.ndarray,
    mask: np.ndarray,
    crop_size: int = 512,
    pad: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Упрощённый crop по маске: находим bbox маски, вырезаем квадрат crop_size с центром в центре bbox.
    img/mask: (H,W) или (H,W,C). Возвращает (crop_img, crop_mask).
    """
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0 or len(xs) == 0:
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
    else:
        cy = int(round(np.mean(ys)))
        cx = int(round(np.mean(xs)))
    half = crop_size // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(img.shape[0], y0 + crop_size)
    x1 = min(img.shape[1], x0 + crop_size)
    if y1 - y0 < crop_size or x1 - x0 < crop_size:
        # pad
        out_img = np.zeros((crop_size, crop_size, 3) if img.ndim == 3 else (crop_size, crop_size), dtype=img.dtype)
        out_mask = np.zeros((crop_size, crop_size), dtype=mask.dtype)
        dy = (crop_size - (y1 - y0)) // 2
        dx = (crop_size - (x1 - x0)) // 2
        out_img[dy : dy + (y1 - y0), dx : dx + (x1 - x0)] = img[y0:y1, x0:x1]
        out_mask[dy : dy + (y1 - y0), dx : dx + (x1 - x0)] = mask[y0:y1, x0:x1]
        return out_img, out_mask
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def face_mask_from_pose_keypoints_simple(
    keypoints: np.ndarray,
    height: int,
    width: int,
    face_keypoint_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Упрощённая маска лица по ключевым точкам (как FaceMaskFromPoseKeypoints).
    keypoints: (K, 3) — x, y, confidence. Стандартные индексы лица в COCO/OpenPose:
    0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear.
    """
    if face_keypoint_indices is None:
        face_keypoint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # face + части головы
    mask = np.zeros((height, width), dtype=np.float32)
    pts = keypoints[keypoints[:, 2] > 0.3][:, :2]
    if len(pts) == 0:
        return mask
    x_min = max(0, int(pts[:, 0].min()) - 20)
    x_max = min(width, int(pts[:, 0].max()) + 20)
    y_min = max(0, int(pts[:, 1].min()) - 20)
    y_max = min(height, int(pts[:, 1].max()) + 20)
    mask[y_min:y_max, x_min:x_max] = 1.0
    return mask


def run_dwpose_on_image(
    img: np.ndarray,
    model_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    DWPose на одном изображении. Возвращает keypoints (K, 3) и опционально визуализацию.
    Требует: controlnet_aux (DWPreprocessor) или onnx runtime.
    """
    try:
        from controlnet_aux import DWposeDetector
    except ImportError:
        raise RuntimeError(
            "For DWPose install: pip install controlnet_aux"
        )
    detector = DWposeDetector()
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    result = detector(pil, detect_resolution=512, image_resolution=min(img.shape[0], img.shape[1]))
    if hasattr(result, "numpy"):
        return np.array(result)
    return np.array(result)


def run_dwpose_on_frames(
    frames: np.ndarray,
    detect_resolution: int = 512,
) -> List[np.ndarray]:
    """DWPose на каждом кадре. frames (T,H,W,C)."""
    try:
        from controlnet_aux import DWposeDetector
    except ImportError:
        raise RuntimeError("pip install controlnet_aux")
    detector = DWposeDetector()
    results = []
    for t in range(frames.shape[0]):
        pil = Image.fromarray((np.clip(frames[t], 0, 1) * 255).astype(np.uint8))
        res = detector(pil, detect_resolution=detect_resolution, image_resolution=min(frames.shape[1], frames.shape[2]))
        results.append(np.array(res))
    return results


def prepare_face_images_from_pose_frames(
    reference_image: np.ndarray,
    pose_frames: np.ndarray,
    width: int = 832,
    height: int = 480,
    face_crop_size: int = 512,
    use_dwpose: bool = True,
) -> np.ndarray:
    """
    Получить face_images как в workflow:
    FaceMaskFromPoseKeypoints → ImageCropByMaskAndResize (512x512).
    Упрощённо: по первому кадру pose делаем маску лица, кроп 512x512, для всех кадров
    используем тот же кроп из reference (или из первого pose).
    """
    if not use_dwpose:
        # Без DWPose: центр кадра как лицо
        c = face_crop_size // 2
        y0 = max(0, height // 2 - c)
        x0 = max(0, width // 2 - c)
        ref_crop = reference_image[y0 : y0 + face_crop_size, x0 : x0 + face_crop_size]
        if ref_crop.shape[0] != face_crop_size or ref_crop.shape[1] != face_crop_size:
            ref_crop = resize_lanczos(reference_image, face_crop_size, face_crop_size)
        return np.tile(ref_crop[np.newaxis, ...], (pose_frames.shape[0], 1, 1, 1))
    try:
        from controlnet_aux import DWposeDetector
    except ImportError:
        return prepare_face_images_from_pose_frames(
            reference_image, pose_frames, width, height, face_crop_size, use_dwpose=False
        )
    detector = DWposeDetector()
    pil_ref = Image.fromarray((np.clip(reference_image, 0, 1) * 255).astype(np.uint8))
    pose_ref = detector(pil_ref, detect_resolution=512, image_resolution=min(height, width))
    # Упрощённо: берём bbox ключевых точек лица и кропаем reference
    pose_arr = np.array(pose_ref)
    if pose_arr.ndim == 2:
        mask = (pose_arr > 0).astype(np.float32)
    else:
        mask = (pose_arr.sum(axis=-1) > 0).astype(np.float32)
    crop_ref, _ = crop_by_mask_center(reference_image, mask, crop_size=face_crop_size)
    if crop_ref.shape[0] != face_crop_size or crop_ref.shape[1] != face_crop_size:
        crop_ref = resize_lanczos(crop_ref, face_crop_size, face_crop_size)
    return np.tile(crop_ref[np.newaxis, ...], (pose_frames.shape[0], 1, 1, 1))


def prepare_mask(
    path: Optional[Union[str, Path]],
    num_frames: int,
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    """Загрузка маски (T,H,W) или (H,W) [0,1]. Если нет — None."""
    if path is None or not Path(path).exists():
        return None
    img = load_image(path)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    if img.shape[0] != height or img.shape[1] != width:
        img = resize_lanczos(img[:, :, np.newaxis], width, height).squeeze(-1)
    if img.ndim == 2:
        img = np.tile(img[np.newaxis, ...], (num_frames, 1, 1))
    return img.astype(np.float32)
