"""Generate a simple app icon for the launcher."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "dist" / "multisearch_icon.ico"
    out.parent.mkdir(parents=True, exist_ok=True)

    size = 256
    img = Image.new("RGBA", (size, size), (13, 20, 40, 255))
    draw = ImageDraw.Draw(img)

    # Border and card body
    draw.rounded_rectangle((18, 18, 238, 238), radius=34, fill=(28, 46, 86, 255), outline=(90, 170, 255, 255), width=6)

    # Search lens
    draw.ellipse((52, 58, 146, 152), outline=(112, 210, 255, 255), width=12)
    draw.line((132, 138, 188, 194), fill=(112, 210, 255, 255), width=12)

    # Mini image blocks
    draw.rounded_rectangle((154, 66, 220, 106), radius=10, fill=(89, 129, 255, 255))
    draw.rounded_rectangle((154, 116, 220, 156), radius=10, fill=(121, 226, 173, 255))

    # Small accent dots
    draw.ellipse((62, 184, 78, 200), fill=(255, 200, 102, 255))
    draw.ellipse((90, 184, 106, 200), fill=(255, 200, 102, 255))
    draw.ellipse((118, 184, 134, 200), fill=(255, 200, 102, 255))

    img.save(out, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"Created icon: {out}")


if __name__ == "__main__":
    main()
