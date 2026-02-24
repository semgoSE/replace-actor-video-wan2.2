# Wan2.2-Animate pipeline

Пайплайн замены актёра / анимации: **preprocess → generation** (опционально: thin, refine, postprocess, upscale). Чистый Python, без ComfyUI.

---

## Структура

```
replace-actor-video/
├── out/                 # выходные видео
├── preprocess/          # результаты препроцессинга (preprocess/<имя>)
├── Wan2.2/              # клон Wan2.2
├── Wan2.2-Animate-14B/  # чекпоинт
├── run_preprocessing.py
├── run_preprocessing_custom.py   # свой препроцесс (DWPose)
├── run_thin_for_generation.py   # 30→15 fps для генерации
├── run_generation.py
├── run_postprocess.py
├── run_upscale.py
├── run_refine.py
├── pipeline.py
└── app_ui.py            # Web-UI
```

---

## Установка

```bash
./install.sh
```

Опции: `--no-checkpoint`, `--no-rife`, `--no-realesrgan`, `--ui`.

Вручную: клонировать Wan2.2, поставить зависимости (см. Wan2.2), скачать чекпоинт `Wan2.2-Animate-14B`, из корня проекта `pip install -r requirements.txt`.

---

## Пайплайн

| Этап | Скрипт | Назначение |
|------|--------|------------|
| 1 | `run_preprocessing.py` | Препроцесс Wan2.2 (DWPose в Wan2.2) |
| 1 (альт.) | `run_preprocessing_custom.py` | Свой препроцесс (dwposes), точки для маски в UI |
| 1.5 | `run_thin_for_generation.py` | Прореживание 30→15 fps (экономия при генерации) |
| 2 | `run_generation.py` | Генерация (Wan2.2 animate-14B) |
| — | `run_refine.py` | Второй проход replace (убрать артефакты) |
| 3 | `run_postprocess.py` | RIFE / ffmpeg: поднять FPS |
| 4 | `run_upscale.py` | SeedVR2 или Real-ESRGAN |

Выход генерации, postprocess, upscale, refine — в **`out/`**.

---

## Быстрый старт

**Полный цикл (preprocess + generation):**

```bash
python run_preprocessing.py --video_path v.mp4 --refer_path r.jpg --save_path preprocess/run1 --replace
python run_generation.py --src_root_path preprocess/run1 --save_file out.mp4 --replace
```

**С прореживанием (30 fps → 15 fps → генерация → RIFE 30 fps):**

```bash
python run_preprocessing.py --video_path v.mp4 --refer_path r.jpg --save_path preprocess/run1 --fps 30 --replace
python run_thin_for_generation.py --input preprocess/run1
python run_generation.py --src_root_path preprocess/run1_gen --save_file out.mp4 --replace
python run_postprocess.py --input out/out.mp4 --output out/out_30fps.mp4 --target_fps 30
```

---

## Этапы (кратко)

### 1. Preprocess (Wan2.2)

```bash
python run_preprocessing.py --video_path VIDEO --refer_path REF --save_path preprocess/NAME [--replace] [--fps 30]
```

Результат в `preprocess/NAME`: `src_pose.mp4`, `src_face.mp4`, `src_ref.png`; при `--replace` ещё `src_bg.mp4`, `src_mask.mp4`.

### 1. Кастомный препроцесс

Без Wan2.2, DWPose через `dwposes`. Точки для маски — во вкладке UI «Точки для SAM2», сохраняются в `preprocess/sam2_points_<имя>.json`.

```bash
pip install dwposes opencv-python imageio imageio-ffmpeg
python run_preprocessing_custom.py --video_path VIDEO --refer_path REF --save_path preprocess/NAME [--replace] [--points_file preprocess/sam2_points_xxx.json]
```

### 1.5. Thin (опц.)

```bash
python run_thin_for_generation.py --input preprocess/NAME
# создаётся preprocess/NAME_gen
```

### 2. Generation

```bash
python run_generation.py --src_root_path preprocess/NAME [--replace] [--save_file out.mp4] [--no_offload]
```

FPS выхода берётся из папки препроцессинга; переопределить: `--fps 30`.

### Refine (опц.)

```bash
python run_refine.py --input out/video.mp4 --refer_path REF --save_file refined.mp4
```

### 3. Postprocess (RIFE / ffmpeg)

```bash
python run_postprocess.py --input IN.mp4 --output out/OUT.mp4 --target_fps 30 [--backend vulkan|cuda|ffmpeg]
```

### 4. Upscale

```bash
python run_upscale.py --input IN.mp4 --output out/OUT.mp4 [--backend realesrgan|seedvr2]
```

---

## Web-UI

```bash
pip install -r requirements-ui.txt
python app_ui.py
```

Открыть http://127.0.0.1:7860. Вкладки: Preprocess (Wan2.2 или кастомный), Точки для SAM2, Thin, Generation, Refine, Postprocess, Upscale, Скачать. Результаты препроцесса — в `preprocess/<имя>`, выходные видео — в `out/`.

---

## Опции (по этапам)

- **Preprocess:** `--resolution "1280 720"`, `--fps 30`, `--replace`, `--use_flux` (Wan2.2).
- **Generation:** `--prompt`, `--sample_steps`, `--sample_guide_scale`, `--sample_shift`, `--frame_num`, `--seed`, `--no_offload`, `--fps`.
- **Postprocess:** нужны ffmpeg; RIFE: [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) (vulkan) или клон Practical-RIFE (cuda).
- **Upscale:** Real-ESRGAN — клон [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) в папку проекта; SeedVR2 — клон [SeedVR](https://github.com/ByteDance-Seed/SeedVR) и чекпоинт.
