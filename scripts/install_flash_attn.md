# Установка flash_attn (предсобранные wheels)

Сборка из исходников часто падает. Ставьте **готовый wheel** под ваши версии PyTorch и CUDA.

## 1. Узнать свои версии

В консоли:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Python:', __import__('sys').version.split()[0])"
```

Запомните: PyTorch (например 2.5.1), CUDA (например 12.1 или None), Python (3.10/3.11/3.12).

## 2. Скачать нужный wheel

### PyTorch 2.10 + CUDA 12.6 + Python 3.12 (Linux x86_64)

Готовый wheel (одна команда):

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu126torch2.10-cp312-cp312-linux_x86_64.whl
```

Если не подойдёт (старый glibc), попробуйте manylinux-вариант:

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu126torch2.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

### Другие версии

Откройте **Wheel Finder** и выберите комбинацию:

- **https://flashattn.dev/** (или https://flashattn.dev/install/prebuilt-wheels)
- Выберите: Platform (Linux/Windows), flash-attn 2.8.x, ваш Python, PyTorch, CUDA.
- Скопируйте команду вида:
  ```bash
  pip install https://.../flash_attn-2.8.3+...whl
  ```

Либо готовые wheels по версиям:
- **https://github.com/mjun0812/flash-attention-prebuild-wheels/releases** — скачайте `.whl` под вашу платформу, Python, torch и CUDA, затем:
  ```bash
  pip install /path/to/downloaded_flash_attn-....whl
  ```

## 3. Установить

Вставьте скопированную команду и выполните:

```bash
pip install https://...  # ваша ссылка с flashattn.dev
```

## 4. Проверить

```bash
python -c "import flash_attn; print('flash_attn ok')"
```

Если видите `flash_attn ok` — всё стоит корректно.

## Если CUDA нет или версия не совпадает

- На машине без GPU или без CUDA flash_attn не нужен — Wan2.2 будет использовать SDPA.
- Если CUDA есть, но версия старая: обновите PyTorch под вашу CUDA (`pip install torch --index-url https://download.pytorch.org/whl/cu121` и т.п.), затем снова подберите wheel под новую версию torch/CUDA на flashattn.dev.
