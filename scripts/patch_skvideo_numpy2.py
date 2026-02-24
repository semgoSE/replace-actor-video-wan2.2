#!/usr/bin/env python3
"""
Патч scikit-video (skvideo) для совместимости с NumPy 2.x.

В NumPy 2 удалены алиасы np.float и np.int — skvideo падает с AttributeError.
Скрипт находит установленный пакет skvideo и заменяет np.float -> float, np.int -> int.

Запуск (из корня проекта или откуда угодно):
  python scripts/patch_skvideo_numpy2.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def find_skvideo_path() -> Path | None:
    try:
        import skvideo
        return Path(skvideo.__path__[0])
    except Exception:
        return None


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    original = text
    # np.float и np.int удалены в NumPy 2
    text = re.sub(r"\bnp\.float\b", "float", text)
    text = re.sub(r"\bnp\.int\b", "int", text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    base = find_skvideo_path()
    if not base or not base.exists():
        print("Пакет skvideo (scikit-video) не найден. Установите: pip install scikit-video", file=sys.stderr)
        return 1
    patched = 0
    for py in base.rglob("*.py"):
        if patch_file(py):
            print("Патч применён:", py)
            patched += 1
    if patched == 0:
        print("Файлы skvideo уже исправлены или не содержат np.float/np.int.")
    else:
        print("Готово. Исправлено файлов:", patched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
