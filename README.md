# Wan2.2-Animate pipeline (чисто Python, без ComfyUI)

Пайплайн в **два этапа**: preprocessing → generation. Без ComfyUI, только Python и официальный репозиторий Wan2.2.

## Требования

1. **Клонировать Wan2.2 и установить зависимости**
   ```bash
   git clone https://github.com/Wan-Video/Wan2.2.git
   cd Wan2.2
   pip install -r requirements.txt
   pip install -r requirements_animate.txt  # для Animate
   ```

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

Режимы:
- **Animate** (по умолчанию): персонаж с референса повторяет движение с видео — `--retarget`
- **Replace**: замена актёра в видео на персонажа с референса — `--replace`

Опции:
- `--wan22_dir Wan2.2` — путь к репозиторию Wan2.2
- `--ckpt_dir Wan2.2-Animate-14B` — путь к чекпоинту
- `--use_flux` — использовать FLUX в препроцессинге

## Этап 2: Generation

Генерация видео по результатам препроцессинга (вызов официального `generate.py` Wan2.2).

```bash
python run_generation.py \
  --src_root_path process_results \
  --output_path output_wanimate.mp4
```

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
