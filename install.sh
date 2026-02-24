#!/usr/bin/env bash
#
# Установка всего нужного для пайплайна (кроме SeedVR2 — на будущее).
# Запуск из корня проекта: ./install.sh [опции]
#
# Опции:
#   --no-checkpoint   не качать Wan2.2-Animate-14B (скачаете вручную)
#   --no-rife         не ставить Practical-RIFE (RIFE на CUDA)
#   --no-realesrgan   не ставить Real-ESRGAN (апскейл)
#   --ui              дополнительно установить зависимости Web-UI (gradio)
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$PROJECT_ROOT"

SKIP_CHECKPOINT=false
SKIP_RIFE=false
SKIP_REALESRGAN=false
WITH_UI=false

for arg in "$@"; do
  case "$arg" in
    --no-checkpoint) SKIP_CHECKPOINT=true ;;
    --no-rife)       SKIP_RIFE=true ;;
    --no-realesrgan) SKIP_REALESRGAN=true ;;
    --ui)            WITH_UI=true ;;
    -h|--help)
      echo "Usage: $0 [--no-checkpoint] [--no-rife] [--no-realesrgan] [--ui]"
      echo "  --no-checkpoint  не качать чекпоинт Wan2.2-Animate-14B"
      echo "  --no-rife        не ставить Practical-RIFE"
      echo "  --no-realesrgan  не ставить Real-ESRGAN"
      echo "  --ui             установить зависимости Web-UI (gradio)"
      exit 0
      ;;
  esac
done

echo "=== replace-actor-video: установка (SeedVR2 не ставим — на будущее) ==="

# --- 1. Wan2.2 ---
if [[ ! -d "$PROJECT_ROOT/Wan2.2" ]]; then
  echo ">>> Клонируем Wan2.2..."
  git clone https://github.com/Wan-Video/Wan2.2.git "$PROJECT_ROOT/Wan2.2"
else
  echo ">>> Wan2.2 уже есть, пропускаем клон."
fi

echo ">>> Зависимости Wan2.2..."
pip install --upgrade pip
pip install torch torchvision
pip install -r "$PROJECT_ROOT/Wan2.2/requirements.txt"
pip install -r "$PROJECT_ROOT/Wan2.2/requirements_animate.txt" || true

# flash_attn: часто не собирается из исходников — см. scripts/install_flash_attn.md
pip install flash-attn --no-build-isolation 2>/dev/null || echo ">>> flash_attn не установился — при необходимости поставьте wheel вручную, см. scripts/install_flash_attn.md"

# --- 2. Чекпоинт Wan2.2-Animate-14B ---
if [[ "$SKIP_CHECKPOINT" == true ]]; then
  echo ">>> Чекпоинт пропущен (--no-checkpoint). Скачайте вручную: huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B"
else
  if [[ ! -d "$PROJECT_ROOT/Wan2.2-Animate-14B" ]] || [[ ! -d "$PROJECT_ROOT/Wan2.2-Animate-14B/process_checkpoint" ]]; then
    echo ">>> Скачиваем Wan2.2-Animate-14B..."
    pip install "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir "$PROJECT_ROOT/Wan2.2-Animate-14B"
  else
    echo ">>> Чекпоинт Wan2.2-Animate-14B уже есть."
  fi
fi

# --- 3. Зависимости оркестрации ---
echo ">>> Зависимости пайплайна..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# --- 4. Practical-RIFE (RIFE на CUDA) ---
if [[ "$SKIP_RIFE" == false ]]; then
  if [[ ! -d "$PROJECT_ROOT/Practical-RIFE" ]]; then
    echo ">>> Клонируем Practical-RIFE..."
    git clone https://github.com/hzwer/Practical-RIFE.git "$PROJECT_ROOT/Practical-RIFE"
  else
    echo ">>> Practical-RIFE уже есть."
  fi
  echo ">>> Зависимости Practical-RIFE (Python 3.12-совместимые)..."
  pip install --upgrade setuptools wheel
  pip install -r "$PROJECT_ROOT/scripts/requirements_practical_rife_py312.txt"
  if python -c "import skvideo" 2>/dev/null; then
    python "$PROJECT_ROOT/scripts/patch_skvideo_numpy2.py" || true
  fi
  echo ">>> Веса RIFE скачайте в Practical-RIFE/train_log/ (см. README Practical-RIFE)."
else
  echo ">>> Practical-RIFE пропущен (--no-rife)."
fi

# --- 5. Real-ESRGAN (апскейл) ---
if [[ "$SKIP_REALESRGAN" == false ]]; then
  if [[ ! -d "$PROJECT_ROOT/Real-ESRGAN" ]]; then
    echo ">>> Клонируем Real-ESRGAN..."
    git clone https://github.com/xinntao/Real-ESRGAN.git "$PROJECT_ROOT/Real-ESRGAN"
  else
    echo ">>> Real-ESRGAN уже есть."
  fi
  echo ">>> Зависимости Real-ESRGAN..."
  pip install -r "$PROJECT_ROOT/Real-ESRGAN/requirements.txt"
  pip install basicsr facexlib gfpgan ffmpeg-python
  cd "$PROJECT_ROOT/Real-ESRGAN" && python setup.py develop && cd "$PROJECT_ROOT"
else
  echo ">>> Real-ESRGAN пропущен (--no-realesrgan)."
fi

# --- 6. Web-UI (опционально) ---
if [[ "$WITH_UI" == true ]]; then
  echo ">>> Устанавливаем Web-UI (gradio)..."
  pip install -r "$PROJECT_ROOT/requirements-ui.txt"
fi

# --- SeedVR2: на будущее ---
echo ""
echo ">>> SeedVR2 не устанавливался (на будущее). Когда понадобится — см. README, раздел «SeedVR2»."
echo ""
echo "=== Готово. Запуск: python run_preprocessing.py ... ; python run_generation.py ... ; python run_postprocess.py ... ; python run_upscale.py --backend realesrgan ... ==="
