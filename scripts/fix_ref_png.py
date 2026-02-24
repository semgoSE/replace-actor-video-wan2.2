#!/usr/bin/env python3
"""
Пересохраняет src_ref.png в process_results в «чистый» PNG.
Исправляет libpng/cv2.imread ошибки (bad parameters to zlib и т.п.).

Запуск:
  python scripts/fix_ref_png.py
  python scripts/fix_ref_png.py --path process_results
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install Pillow: pip install Pillow")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("process_results"), help="Папка с process_results")
    args = parser.parse_args()
    ref_path = args.path / "src_ref.png"
    if not ref_path.exists():
        raise SystemExit(f"Файл не найден: {ref_path}")
    # Загружаем через PIL (часто читает «битые» PNG лучше) и сохраняем заново
    img = Image.open(ref_path).convert("RGB")
    out_path = ref_path.with_suffix(".fixed.png")
    img.save(out_path, "PNG", compress_level=6)
    # Заменяем оригинал
    ref_path.unlink()
    out_path.rename(ref_path)
    print(f"Пересохранён: {ref_path}")

if __name__ == "__main__":
    main()
