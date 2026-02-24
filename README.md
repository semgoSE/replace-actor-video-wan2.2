# Wan2.2-Animate pipeline (чисто Python, без ComfyUI)

Пайплайн в **два этапа**: preprocessing → generation. Без ComfyUI, только Python и официальный репозиторий Wan2.2.

## Структура проекта

Все зависимости кладутся **в папку проекта** (как Wan2.2). Запуск скриптов — из корня проекта.

```
replace-actor-video/          # корень (здесь запускаете python run_*.py)
├── Wan2.2/                   # клон Wan2.2
├── Wan2.2-Animate-14B/       # чекпоинт (process_checkpoint внутри)
├── SeedVR/                   # опционально: апскейл SeedVR2
├── Real-ESRGAN/              # опционально: апскейл Real-ESRGAN
├── Practical-RIFE/           # опционально: RIFE на CUDA (--backend cuda)
├── run_preprocessing.py
├── run_generation.py
├── run_postprocess.py
├── run_upscale.py
├── pipeline.py
├── config.py
└── ...
```

По умолчанию скрипты ищут `Wan2.2/`, `SeedVR/`, `Real-ESRGAN/`, `Practical-RIFE/` в текущей директории; пути можно задать через `--wan22_dir`, `--seedvr_dir`, `--realesrgan_dir`, `--rife_pytorch_dir` или переменные окружения.

## Требования

1. **Клонировать Wan2.2 в папку проекта и установить зависимости**
   ```bash
   cd /path/to/replace-actor-video   # корень проекта
   git clone https://github.com/Wan-Video/Wan2.2.git
   cd Wan2.2
   pip install torch torchvision   # сначала PyTorch!
   pip install -r requirements.txt
   pip install -r requirements_animate.txt  # для Animate
   cd ..
   ```
   Если **flash_attn** не собирается — предсобранный wheel: [scripts/install_flash_attn.md](scripts/install_flash_attn.md). Для **режима replace**: `pip install sam2` или `pip install -r Wan2.2/requirements_animate.txt` из корня проекта.

2. **Скачать модель Wan2.2-Animate-14B** (в корень проекта)
   ```bash
   pip install "huggingface_hub[cli]"
   huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B
   ```
   Или через ModelScope: `modelscope download Wan-AI/Wan2.2-Animate-14B --local_dir ./Wan2.2-Animate-14B`.

3. **Зависимости оркестрации**
   ```bash
   pip install -r requirements.txt
   ```

**Или одной командой (всё кроме SeedVR2):** из корня проекта запустите скрипт установки:
   ```bash
   ./install.sh
   ```
   Опции: `--no-checkpoint` (не качать чекпоинт), `--no-rife`, `--no-realesrgan`, `--ui` (добавить Gradio для Web-UI). SeedVR2 скрипт не ставит — см. раздел ниже, когда понадобится.

## Этап 1: Preprocessing

Подготовка данных из видео и референсного изображения (вызов официального `preprocess_data.py` Wan2.2).

```bash
python run_preprocessing.py \
  --video_path path/to/dance.mp4 \
  --refer_path path/to/face.jpg \
  --save_path process_results \
  --resolution "1280 720"
```

### Что получается после preprocess (в `--save_path`)

В папке `process_results` (или любой указанной в `--save_path`) появляются файлы:

**Режим Animate** (по умолчанию, `--retarget`):

| Файл | Описание |
|------|----------|
| `src_ref.png` | Копия референсного изображения (ваше фото персонажа) |
| `src_face.mp4` | Видео из кропов лица по кадрам (512×512), для контроля лица |
| `src_pose.mp4` | Видео с нарисованной позой (conditioning) под модель |

При `--use_flux` дополнительно: `refer_edit.png`, `tpl_edit.png` (отредактированные кадры для retarget).

**Режим Replace** (`--replace`):

| Файл | Описание |
|------|----------|
| `src_ref.png` | Референсное изображение |
| `src_face.mp4` | Видео кропов лица |
| `src_pose.mp4` | Conditioning позы |
| `src_bg.mp4` | Фон (видео без персонажа) |
| `src_mask.mp4` | Маска персонажа по кадрам |

Эту папку целиком передают в этап 2: `--src_root_path process_results`.

### Маска: ComfyUI (wanvideo_WanAnimate) vs Wan2.2

В твоём **ComfyUI workflow** маска строилась так:

1. **PointsEditor** — ты (или скрипт) задаёшь точки на изображении по персонажу.
2. **Sam2Segmentation** — SAM2 по этим точкам вырезает регион → маска повторяет **контур** человека (не квадрат).
3. **GrowMask** — расширение маски на **10 px** (`widgets_values: 10, true`).
4. **BlockifyMask** — выравнивание по блокам **32 px** (`widgets_values: 32, "cpu"`).

Итого: маска = **точечная сегментация SAM2** (по кликам) → contour → grow 10 → blockify 32.

В **Wan2.2 replace** маска строится по-другому:

- Точки для SAM2 берутся **автоматически** из позы (ключевые точки тела: нос, плечи, бёдра, колени и т.д.), без ручных кликов.
- Дальше идёт расширение маски (`iterations`, `k`, `w_len`, `h_len`).

Из-за этого маска часто получается **квадратной/прямоугольной**: автоматические ключевые точки задают грубый регион, SAM2 его достраивает, и после dilation форма может быть близка к bbox. В ComfyUI контур был точнее, потому что точки задавались вручную по силуэту.

**Если нужна маска «как в ComfyUI»** (по контуру): в текущем Wan2.2 препроцессинге такого режима нет. Варианты — подбирать `--iterations`/`--k` в препроцессинге или доработать пайплайн: отдельный шаг SAM2 по точкам (из файла или по центру лица/тела) → grow 10 → blockify 32 и подмена `src_mask.mp4`.

Режимы:
- **Animate** (по умолчанию): персонаж с референса повторяет движение с видео — `--retarget`
- **Replace**: замена актёра в видео на персонажа с референса — `--replace`

Опции:
- `--wan22_dir Wan2.2` — путь к репозиторию Wan2.2
- `--ckpt_dir Wan2.2-Animate-14B` — путь к чекпоинту
- `--use_flux` — использовать FLUX в препроцессинге

## Этап 2: Generation

Генерация видео по результатам препроцессинга (вызов официального `generate.py` Wan2.2).

### Steps / CFG / Prompts (дефолты Wan2.2 animate-14B)

| Параметр | Аргумент | Ориентир | Описание |
|----------|----------|----------|----------|
| **Prompt** | `--prompt` | дефолт: "视频中的人在做动作" | Текстовое описание сцены |
| **Steps** | `--sample_steps` | 30–40 | Число шагов сэмплинга |
| **CFG** | `--sample_guide_scale` | 2–4.5 | Classifier-free guidance scale |
| **Shift** | `--sample_shift` | 3–5 | Flow shift для сэмплера |
| **Кадры** | `--frame_num` | из конфига (4n+1) | Количество кадров |
| **Seed** | `--seed` | 42 | Базовый seed (`--base_seed` в Wan2.2) |

Если не передавать `--prompt` / `--sample_steps` / `--sample_guide_scale` — используются дефолты из конфига Wan2.2 для задачи animate-14B.

```bash
python run_generation.py \
  --src_root_path process_results \
  --output_path output_wanimate.mp4
```

С явными steps/CFG/prompt:
```bash
python run_generation.py \
  --src_root_path process_results \
  --prompt "a person dancing in a studio, natural lighting" \
  --sample_steps 30 \
  --sample_guide_scale 2.5 \
  --sample_shift 5 \
  --save_file my_output.mp4
```

**Максимальная скорость** (если хватает VRAM, обычно 24GB+): модель не выгружается на CPU — генерация быстрее:
```bash
python run_generation.py --src_root_path process_results --no_offload --save_file out.mp4
```
Полный пайплайн с `--no_offload`: `python pipeline.py --video_path v.mp4 --refer_path r.jpg --no_offload`. При нескольких GPU: `--multi_gpu --nproc_per_node N`.

### Уровень 1–2: TF32, Flash Attention, torch.compile

| Оптимизация | Статус |
|-------------|--------|
| **TF32** | Включён в `Wan2.2/generate.py`: при запуске генерации вызывается `torch.set_float32_matmul_precision("high")` — ускорение matmul на GPU Ampere+ (A100, RTX 30xx+). |
| **Flash Attention 3** | Уже в коде Wan2.2: в `wan/modules/attention.py` при наличии `flash_attn_interface` используется FA3, иначе FA2. Чтобы использовать FA3, нужна сборка flash_attn с поддержкой FA3 (зависит от версии и колеса). |
| **torch.compile** | Используется только в **препроцессинге** (SAM2 image encoder). В контуре **генерации** (animate DiT) в Wan2.2 не вызывается; добавление потребовало бы правок внутри Wan2.2 (offload и динамика усложняют compile). |

Итого: TF32 уже включён; Flash Attention 2 (или 3, если установлен соответствующий wheel) используется; torch.compile для этапа генерации в текущем Wan2.2 нет.

### Почему может быть медленнее ComfyUI (даже с --no_offload)

Пайплайн вызывает **официальный** `Wan2.2/generate.py` — тот же движок, но без оптимизаций, которые могут быть в нодах ComfyUI (свой бэкенд, компиляция, батчинг). Чтобы приблизить скорость:

- Используй **меньше шагов**: дефолт Wan2.2 для animate — **20** шагов. В твоём ComfyUI было 30. Для скорости явно передавай `--sample_steps 20` (качество чуть ниже, быстрее).
- Обязательно **--no_offload** при достаточном VRAM.
- Если в ComfyUI было другое разрешение или длина клипа — при равных условиях время будет ближе.

### Идеи: ускорение в 2–3 раза

| Способ | Как | Ориентир | Компромисс |
|--------|-----|----------|------------|
| **Меньше шагов** | `--sample_steps 10` или `--sample_steps 12` | ~2× при 20→10 шагов | Качество может упасть; стоит сравнивать на своём материале. |
| **Ниже разрешение** | Препроцессинг и генерация в меньшем разрешении: `--resolution "960 540"` (этап 1) и тот же размер в Wan2.2 | ~1.5–2.5× | Меньше деталей, ниже разрешение видео. |
| **Два GPU** | `--multi_gpu --nproc_per_node 2 --no_offload` | до ~2× | Нужны 2 GPU. |
| **Другой сэмплер** | `--sample_solver dpm++` (по умолч. `unipc`); иногда при меньшем числе шагов даёт приемлемое качество быстрее | зависит от сцены | Можно пробовать с `--sample_steps 12`. |
| **--no_offload** | Модель не сбрасывать на CPU | заметно быстрее при достаточном VRAM | Нужно ~24GB+ VRAM. |

Комбо для максимальной скорости (качество ниже): `--no_offload --sample_steps 10 --sample_solver dpm++`. Для препроцессинга — то же разрешение, что планируешь для генерации (например 960×540).

Для режима замены актёра:
```bash
python run_generation.py \
  --src_root_path process_results \
  --replace \
  --use_relighting_lora
```

## Полный пайплайн (оба этапа подряд)

```bash
python pipeline.py \
  --video_path path/to/dance.mp4 \
  --refer_path path/to/face.jpg \
  --save_path process_results \
  --output_path output_wanimate.mp4
```

Режим replace:
```bash
python pipeline.py --video_path v.mp4 --refer_path r.jpg --replace --use_relighting_lora
```

## Refine (опционально): второй проход replace — убрать артефакты

После первого replace часто остаются артефакты (остатки одежды, границы маски). **Refine** — это тот же пайплайн (preprocess + generation в режиме replace), но входом служит уже сгенерированное видео. Модель перерисовывает персонажа и может уменьшить артефакты.

**Вход:** видео после первого replace (или generation). **Референс:** тот же, что в первом replace.

```bash
python run_refine.py \
  --input Wan2.2/out/output.mp4 \
  --refer_path path/to/ref.jpg \
  --save_file refined.mp4
```

Опции: `--save_path` (папка для препроцессинга refine, по умолч. `refine_process`), `--resolution`, `--fps`, `--prompt`, `--sample_steps`, `--sample_guide_scale`, `--sample_shift`, `--seed`, `--no_offload`.

Порядок: **preprocess → generation (replace) → [refine] → postprocess (RIFE) → upscale**.

## Этап 3 (опционально): Post-processing — RIFE (16 fps → 30 fps)

Отдельный этап после animate/replace: повышение FPS за счёт дорисовки кадров (интерполяция). Удобно, если препроцессинг был с `--fps 16`.

**RIFE (рекомендуется):** скачать [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan/releases), положить бинарник в `PATH` или указать `--rife_bin`:

```bash
python run_postprocess.py --input Wan2.2/out/my_540.mp4 --output out_30fps.mp4 --target_fps 30
python run_postprocess.py --input video_16fps.mp4 --output video_30fps.mp4 --rife_bin /path/to/rife-ncnn-vulkan
```

**Без RIFE** (только ffmpeg, качество хуже):

```bash
python run_postprocess.py --input video_16fps.mp4 --output video_30fps.mp4 --fallback_ffmpeg
```

**RIFE на CUDA/PyTorch (серверные GPU, H100/A100):** если Vulkan недоступен, клонируйте Practical-RIFE в папку проекта:

```bash
cd /path/to/replace-actor-video
git clone https://github.com/hzwer/Practical-RIFE
cd Practical-RIFE
pip install --upgrade pip setuptools wheel
pip install -r ../scripts/requirements_practical_rife_py312.txt
# После установки scikit-video под NumPy 2.x один раз:
python ../scripts/patch_skvideo_numpy2.py
# Скачайте модель в train_log/ (см. README Practical-RIFE).
cd ..
python run_postprocess.py --input video.mp4 --output out_30fps.mp4 --backend cuda
# по умолчанию используется ./Practical-RIFE; или --rife_pytorch_dir ./Practical-RIFE
```

**Серверные GPU без RIFE:** `--backend ffmpeg` (или `--fallback_ffmpeg`) — только ffmpeg minterpolate, без доп. установок.

Требуется: **ffmpeg** и **ffprobe**. Vulkan: [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan/releases). CUDA: клон [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) + веса.

## Этап 4 (опционально): Upscale — SeedVR2 или Real-ESRGAN

Следующий этап после поднятия FPS: апскейл видео. Два варианта:

| Бэкенд | Описание | Когда использовать |
|--------|----------|---------------------|
| **SeedVR2** | Одношаговое восстановление до 720p/1080p/2K ([ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR), ICLR 2026) | Лучшее качество, тяжёлая модель (H100/много VRAM) |
| **Real-ESRGAN** | Апскейл 2×/4× по кадрам ([xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)) | Легче, быстрее, один GPU |

### SeedVR2

**Установка SeedVR в папку проекта:**

```bash
cd /path/to/replace-actor-video
git clone https://github.com/ByteDance-Seed/SeedVR
cd SeedVR
conda create -n seedvr python=3.10 -y && conda activate seedvr
pip install -r requirements.txt
pip install flash_attn==2.5.9.post1 --no-build-isolation
# apex: см. README SeedVR (pre-built whl для Python 3.10 / CUDA 12.1)
cd ..
```

Чекпоинт SeedVR2-3B (или 7B) с [Hugging Face](https://huggingface.co/models?other=seedvr) положить в `SeedVR/ckpts/` (например `seedvr2_ema_3b.pth`).

**Запуск** (из корня проекта; по умолчанию используется `./SeedVR`):

```bash
# 720p
python run_upscale.py --input out_30fps.mp4 --output out_720p.mp4

# 1080p
python run_upscale.py --input out_30fps.mp4 --output out_1080p.mp4 --res_h 1080 --res_w 1920
```

Опции: `--res_h` / `--res_w`, `--num_gpus` / `--sp_size`, `--no_audio`.

### Real-ESRGAN

**Установка в папку проекта:**

```bash
cd /path/to/replace-actor-video
git clone https://github.com/xinntao/Real-ESRGAN
cd Real-ESRGAN
pip install -r requirements.txt
pip install basicsr facexlib gfpgan ffmpeg-python
python setup.py develop
cd ..
# веса качаются автоматически при первом запуске (или положите в Real-ESRGAN/weights/)
```

**Запуск** (из корня проекта; по умолчанию используется `./Real-ESRGAN`):

```bash
# 2× апскейл, модель realesr-general-x4v3
python run_upscale.py --input out_30fps.mp4 --output out_2x.mp4 --backend realesrgan --outscale 2

# 4×, модель для фото
python run_upscale.py --input out_30fps.mp4 --output out_4x.mp4 --backend realesrgan --outscale 4 --model_name RealESRGAN_x4plus

# аниме-модель
python run_upscale.py --input out_30fps.mp4 --output out_2x.mp4 --backend realesrgan --model_name realesr-animevideov3
```

Опции: `--model_name` (realesr-general-x4v3 | RealESRGAN_x4plus | RealESRGAN_x4plus_anime_6B | realesr-animevideov3), `--outscale` 2 или 4, `--tile` (при OOM), `--face_enhance` (GFPGAN), `--fp32`.

Порядок пайплайна: **preprocess → generation → postprocess (RIFE 16→30 fps) → upscale (SeedVR2 или Real-ESRGAN)**.

## Файлы

| Файл | Назначение |
|------|------------|
| `config.py` | Конфиг пайплайна (пути, флаги) |
| `run_preprocessing.py` | **Этап 1**: вызов Wan2.2 preprocess_data.py |
| `run_generation.py` | **Этап 2**: вызов Wan2.2 generate.py (animate-14B) |
| `run_refine.py` | **Refine** (опц.): второй проход replace для уменьшения артефактов |
| `run_postprocess.py` | **Этап 3** (опц.): RIFE или ffmpeg — повышение FPS (16→30) |
| `run_upscale.py` | **Этап 4** (опц.): SeedVR2 или Real-ESRGAN — апскейл после RIFE |
| `pipeline.py` | Запуск обоих этапов подряд |
| `app_ui.py` | **Web-UI** (Gradio): вкладки для всех этапов, запуск из браузера |
| `preprocess.py` | Вспомогательные функции (загрузка/resize изображений и видео) |

ComfyUI не используется.

### Web-UI (опционально)

Интерфейс в браузере для всех этапов (Preprocess → Generation → Postprocess → Upscale):

```bash
pip install -r requirements-ui.txt
python app_ui.py
```

Откроется страница (обычно http://127.0.0.1:7860). Каждый этап — своя вкладка; UI только вызывает существующие скрипты `run_*.py`, запущенная генерация в другом терминале не затрагивается.

python run_generation.py   --src_root_path process_results   --prompt "a person dancing in a studio, natural lighting"   --sample_steps 30   --sample_guide_scale 2.5   --sample_shift 5   --save_file my_output.mp4 --replace


python run_preprocessing.py --replace --video_path ./inputs/example.mp4 --refer_path ./inputs/ref.png --save_path ./preprocess/1280_720 --resolution "1280 720"
python run_generation.py --src_root_path ./preprocess/1280_720 --no_offload --use_relighting_lora --prompt "a person dancing in a studio, natural lighting" --sample_steps 30 --sample_shift 5 --save_file ./out/my_30_648.mp4 --replace

python run_postprocess.py --input ./Wan2.2/out/my_30_648.mp4 --output ./Wan2.2/out/my_30_648_30fps.mp4 --backend cuda