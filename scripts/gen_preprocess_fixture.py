"""一次性 script：產生 preprocess 整合測試用的合成 panel 圖。

執行方式（從專案根目錄）：
    python scripts/gen_preprocess_fixture.py
"""
import cv2
import numpy as np
from pathlib import Path

out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "preprocess"
out.mkdir(parents=True, exist_ok=True)

img = np.zeros((1000, 1500), np.uint8)
cv2.rectangle(img, (150, 100), (1350, 900), 180, -1)
# 加雜訊讓 Otsu 工作
rng = np.random.default_rng(42)
noise = rng.integers(0, 20, img.shape, dtype=np.uint8)
img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

dest = out / "synthetic_panel.png"
cv2.imwrite(str(dest), img)
print(f"Fixture written: {dest}")
