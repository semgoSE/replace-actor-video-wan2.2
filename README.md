# Wan2.2-Animate pipeline (чисто Python, без ComfyUI)

Пайплайн в **два этапа**: preprocessing → generation. Без ComfyUI, только Python и официальный репозиторий Wan2.2.

## Требования

1. **Клонировать Wan2.2 и установить зависимости**
   ```bash
   git clone https://github.com/Wan-Video/Wan2.2.git
   cd Wan2.2
   pip install torch torchvision   # сначала PyTorch!
   pip install -r requirements.txt
   pip install -r requirements_animate.txt  # для Animate
   ```
   Если **flash_attn** не собирается из исходников — ставьте предсобранный wheel: см. [scripts/install_flash_attn.md](scripts/install_flash_attn.md). Кратко: узнайте версии PyTorch и CUDA (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`), откройте https://flashattn.dev/ , выберите подходящий wheel и выполните выданную команду `pip install https://...`.
   Для **режима replace** препроцессингу нужен **sam2** (Segment Anything 2): `pip install sam2` или `pip install -r Wan2.2/requirements_animate.txt` из корня Wan2.2.

2. **Скачать модель Wan2.2-Animate-14B**
   ```bash
   pip install "huggingface_hub[cli]"
   huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B
   ```
   Или через ModelScope:
   ```bash
   pip install modelscope
   modelscope download Wan-AI/Wan2.2-Animate-14B --local_dir ./Wan2.2-Animate-14B
   ```

3. **Установить зависимости этого репозитория** (оркестрация)
   ```bash
   pip install -r requirements.txt
   ```

Структура путей по умолчанию:
- `Wan2.2/` — корень репозитория Wan2.2
- `Wan2.2-Animate-14B/` — чекпоинт (внутри должна быть папка `process_checkpoint`)

### Если модели уже есть в ComfyUI

Модели для этого пайплайна берутся из **Wan2.2-Animate-14B** (скачка с Hugging Face / ModelScope). ComfyUI использует свою структуру папок; наш пайплайн вызывает скрипты Wan2.2 и ожидает чекпоинт в формате Wan2.2.

Можно не качать заново, а скопировать/ symlink из ComfyUI:

1. **Чекпоинт Wan2.2-Animate** — в ComfyUI обычно лежит в `ComfyUI/models/checkpoints/` (например `wan2.2_animate_14B_bf16.safetensors`). Для Wan2.2 нужна папка с разложенной моделью (как после `huggingface-cli download Wan-AI/Wan2.2-Animate-14B`). Либо скачайте Wan2.2-Animate-14B отдельно, либо посмотрите в [документации Wan2.2](https://github.com/Wan-Video/Wan2.2), как собрать чекпоинт из одного .safetensors.

2. **Препроцессинг** (`process_checkpoint`) — в составе Wan2.2-Animate-14B с Hugging Face есть папка `process_checkpoint` (pose2d, det, sam2, опционально FLUX). Её нет в стандартных папках ComfyUI; её нужно взять из скачанного Wan2.2-Animate-14B или из репозитория Wan2.2 examples.

Итого: для этого репозитория модели должны быть в `Wan2.2-Animate-14B/` (и репозиторий `Wan2.2/`). Если хотите использовать те же веса, что в ComfyUI, скопируйте нужные файлы в такую структуру или укажите пути через `--ckpt_dir` / `--wan22_dir`.

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

## Файлы

| Файл | Назначение |
|------|------------|
| `config.py` | Конфиг пайплайна (пути, флаги) |
| `run_preprocessing.py` | **Этап 1**: вызов Wan2.2 preprocess_data.py |
| `run_generation.py` | **Этап 2**: вызов Wan2.2 generate.py (animate-14B) |
| `pipeline.py` | Запуск обоих этапов подряд |
| `preprocess.py` | Вспомогательные функции (загрузка/resize изображений и видео) |

ComfyUI не используется.
