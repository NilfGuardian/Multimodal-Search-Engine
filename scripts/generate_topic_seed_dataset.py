"""Generate a topic-labeled starter image dataset for first-run usability."""

from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw

TOPICS = [
    "car",
    "dog",
    "cat",
    "food",
    "flower",
    "space",
    "football",
    "basketball",
    "robot",
    "fashion",
    "bridge",
    "museum",
]


def _topic_color(topic: str, seed: int) -> tuple[int, int, int]:
    random.seed(hash(topic) + seed)
    return (
        random.randint(20, 235),
        random.randint(20, 235),
        random.randint(20, 235),
    )


def _draw_pattern(draw: ImageDraw.ImageDraw, w: int, h: int, seed: int) -> None:
    random.seed(seed)
    for _ in range(10):
        x1 = random.randint(0, w - 50)
        y1 = random.randint(0, h - 50)
        x2 = random.randint(x1 + 20, min(w, x1 + 220))
        y2 = random.randint(y1 + 20, min(h, y1 + 180))
        c = (random.randint(30, 250), random.randint(30, 250), random.randint(30, 250))
        if random.random() > 0.5:
            draw.rectangle((x1, y1, x2, y2), outline=c, width=4)
        else:
            draw.ellipse((x1, y1, x2, y2), outline=c, width=4)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    per_topic = 10
    generated = 0

    for topic in TOPICS:
        for idx in range(1, per_topic + 1):
            filename = f"{topic}_seed_{idx:03d}.jpg"
            target = images_dir / filename
            if target.exists():
                continue

            w, h = 640, 480
            img = Image.new("RGB", (w, h), _topic_color(topic, idx))
            draw = ImageDraw.Draw(img)
            _draw_pattern(draw, w, h, seed=idx * 101)
            draw.text((24, 24), f"{topic} {idx}", fill=(255, 255, 255))
            img.save(target, format="JPEG", quality=90)
            generated += 1

    print(f"Generated {generated} starter images in {images_dir}")


if __name__ == "__main__":
    main()
