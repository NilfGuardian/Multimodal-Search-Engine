"""Generate a local starter dataset of 50 synthetic images.

Run:
    python scripts/generate_sample_images.py
"""

from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw


def _make_image(idx: int, out_path: Path) -> None:
    random.seed(idx)
    w, h = 640, 480
    bg = (random.randint(20, 230), random.randint(20, 230), random.randint(20, 230))
    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)

    for _ in range(12):
        x1 = random.randint(0, w - 40)
        y1 = random.randint(0, h - 40)
        x2 = random.randint(x1 + 20, min(w, x1 + 220))
        y2 = random.randint(y1 + 20, min(h, y1 + 220))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if random.random() > 0.5:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        else:
            draw.ellipse([x1, y1, x2, y2], outline=color, width=4)

    draw.text((20, 20), f"sample_{idx:03d}", fill=(255, 255, 255))
    img.save(out_path, format="JPEG", quality=92)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 51):
        target = images_dir / f"sample_{i:03d}.jpg"
        _make_image(i, target)

    print(f"Generated 50 images in: {images_dir}")


if __name__ == "__main__":
    main()
