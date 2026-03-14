"""Crawl a diverse public-domain style image corpus from Wikimedia Commons.

This is a practical web-crawler style dataset bootstrap for local semantic search.

Run:
    python scripts/crawl_wikimedia_dataset.py --per-topic 40 --max-images 1200
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.parse
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
FALLBACK_TOPIC_IMG = "https://loremflickr.com/1024/768/{topic}?lock={idx}"

REQUEST_HEADERS = {
    "User-Agent": "MultimodalSearchEngineCollegeProject/1.0 (educational crawler)",
    "Accept": "application/json,text/html,*/*",
}

TOPICS = [
    "cars",
    "motorcycle",
    "airplane",
    "train",
    "ship",
    "bicycle",
    "dog",
    "cat",
    "bird",
    "horse",
    "wildlife",
    "mountains",
    "beach",
    "forest",
    "sunset",
    "city skyline",
    "street",
    "architecture",
    "bridge",
    "castle",
    "food",
    "fruits",
    "vegetables",
    "coffee",
    "sports",
    "football",
    "basketball",
    "tennis",
    "computer",
    "smartphone",
    "robot",
    "space",
    "planet",
    "flower",
    "tree",
    "art",
    "painting",
    "museum",
    "fashion",
    "people portrait",
]


def _iter_topic_urls(topic: str, per_topic: int) -> Iterable[tuple[str, str]]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {topic}",
        "gsrlimit": str(per_topic),
        "gsrnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "768",
        "format": "json",
    }
    response = requests.get(COMMONS_API, params=params, headers=REQUEST_HEADERS, timeout=40)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})

    for page in pages.values():
        title = page.get("title", "File:unknown")
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if url:
            yield title, url


def _iter_fallback_urls(topic: str, per_topic: int) -> Iterable[tuple[str, str]]:
    """Fallback topic-based source when Wikimedia API is blocked."""
    topic_slug = urllib.parse.quote(topic.replace(" ", ","))
    for idx in range(1, per_topic + 1):
        yield (
            f"fallback_{topic}_{idx}",
            FALLBACK_TOPIC_IMG.format(topic=topic_slug, idx=idx),
        )


def _safe_name(topic: str, file_title: str, index: int, content_type: str | None) -> str:
    ext = ".jpg"
    if content_type:
        c = content_type.lower()
        if "png" in c:
            ext = ".png"
        elif "webp" in c:
            ext = ".webp"
        elif "jpeg" in c or "jpg" in c:
            ext = ".jpg"

    title_clean = (
        file_title.replace("File:", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    topic_clean = topic.replace(" ", "_")
    return f"{topic_clean}_{index:05d}_{title_clean[:60]}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Wikimedia Commons images for broad search coverage")
    parser.add_argument("--per-topic", type=int, default=35, help="Target images fetched per topic")
    parser.add_argument("--max-images", type=int, default=1000, help="Overall cap across all topics")
    parser.add_argument("--sleep", type=float, default=0.15, help="Pause between downloads in seconds")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    for topic in TOPICS:
        if downloaded >= args.max_images:
            break

        print(f"[crawl] Topic: {topic}")
        try:
            candidates = list(_iter_topic_urls(topic, args.per_topic))
            if not candidates:
                raise RuntimeError("No candidates from Wikimedia")
        except Exception as exc:  # noqa: BLE001
            print(f"[crawl] Wikimedia failed for '{topic}', using fallback source: {exc}")
            candidates = list(_iter_fallback_urls(topic, args.per_topic))

        for i, (title, url) in enumerate(candidates, start=1):
            if downloaded >= args.max_images:
                break

            try:
                resp = requests.get(url, headers=REQUEST_HEADERS, timeout=40)
                resp.raise_for_status()
                content = resp.content

                # Validate image bytes before saving.
                with Image.open(io.BytesIO(content)) as img:
                    img.verify()

                filename = _safe_name(topic, title, i, resp.headers.get("Content-Type"))
                out_path = images_dir / filename
                if out_path.exists():
                    skipped += 1
                    continue
                out_path.write_bytes(content)
                downloaded += 1

                if downloaded % 50 == 0:
                    print(f"[crawl] Downloaded {downloaded} images...")
            except Exception as exc:  # noqa: BLE001
                # Secondary fallback to ensure coverage even when primary source is rate-limited.
                try:
                    fallback_url = FALLBACK_TOPIC_IMG.format(
                        topic=urllib.parse.quote(topic.replace(" ", ",")),
                        idx=i,
                    )
                    fb = requests.get(fallback_url, headers=REQUEST_HEADERS, timeout=40)
                    fb.raise_for_status()
                    content = fb.content
                    with Image.open(io.BytesIO(content)) as img:
                        img.verify()

                    filename = _safe_name(topic, f"fallback_{topic}_{i}", i, fb.headers.get("Content-Type"))
                    out_path = images_dir / filename
                    if out_path.exists():
                        skipped += 1
                    else:
                        out_path.write_bytes(content)
                        downloaded += 1
                except Exception as fallback_exc:  # noqa: BLE001
                    skipped += 1
                    print(f"[crawl] Skip {title}: {exc} | fallback failed: {fallback_exc}")
            finally:
                time.sleep(args.sleep)

    print(f"[crawl] Done. Downloaded={downloaded}, Skipped={skipped}, Output={images_dir}")


if __name__ == "__main__":
    main()
