#!/usr/bin/env python3
"""Печатает версии PyTorch и CUDA — нужно для выбора предсобранного wheel flash_attn на flashattn.dev"""
import sys
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda or "none")
    print("Python:", f"{sys.version_info.major}.{sys.version_info.minor}")
except ImportError:
    print("PyTorch не установлен. Сначала: pip install torch torchvision")
    sys.exit(1)
