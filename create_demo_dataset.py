"""Create a small synthetic dataset for pipeline/UI testing only.

This is not a medically valid training dataset. Replace dataset/ with real,
properly licensed face images before using SkinAura for meaningful predictions.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

import config


rng = np.random.default_rng(42)
styles = {
    "dry": ((196, 151, 130), 24, 5),
    "normal": ((198, 164, 146), 10, 2),
    "oily": ((190, 158, 134), 16, 14),
}

for label, (base_color, texture_strength, shine_strength) in styles.items():
    output_dir = os.path.join(config.TRAIN_DIR, label)
    os.makedirs(output_dir, exist_ok=True)
    for index in range(36):
        height, width = 128, 128
        image = np.zeros((height, width, 3), dtype=np.float32)
        noise = rng.normal(0, texture_strength, (height, width, 1))
        image[:] = np.array(base_color, dtype=np.float32)
        image += noise

        if label == "oily":
            center_x = int(rng.integers(42, 88))
            center_y = int(rng.integers(42, 88))
            yy, xx = np.ogrid[:height, :width]
            shine = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / 900) * shine_strength
            image += shine[..., None]
        elif label == "dry":
            for _ in range(10):
                x = int(rng.integers(5, 120))
                y = int(rng.integers(5, 120))
                image[max(0, y - 1):y + 1, max(0, x - 5):x + 5] -= 12

        image = np.clip(image, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image, mode="RGB")
        draw = ImageDraw.Draw(pil_image)
        draw.ellipse((28, 24, 100, 112), outline=(120, 95, 86), width=1)
        pil_image.save(os.path.join(output_dir, f"demo_{index:03d}.png"))

print(f"Created synthetic demo images in {config.TRAIN_DIR}")
print("Replace this dataset with real, licensed face images for meaningful predictions.")
