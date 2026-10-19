"""OpenStax figures: the training set for the LoRA and the reference style.

Each figure keeps its title, alt text, caption, source file and page, so every
training image can be traced back to the book. Figures credited to anyone other
than OpenStax (photos, micrographs) are dropped, because their rights can differ
from the book's CC BY-NC-SA licence."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from PIL import Image
from pydantic import BaseModel

from ..config import ImagesCfg, SourceCfg
from ..ingest import CNX, _render, _title, chapter_modules, fetch, page_url, raw_url


class Figure(BaseModel):
    id: str
    chapter: int
    section: str
    module_id: str
    title: str
    alt: str
    caption: str
    src: str  # path inside the source repo
    file: str  # local, padded 512 px copy, relative to data/figures
    url: str
    width: int
    height: int


def _figures_in(xml: bytes) -> list[tuple[ET.Element, str, str, str]]:
    root = ET.fromstring(xml)
    out = []
    for fig in root.iter(f"{{{CNX}}}figure"):
        media = fig.find(f"{{{CNX}}}media")
        image = media.find(f"{{{CNX}}}image") if media is not None else None
        if image is None:
            continue
        caption_el = fig.find(f"{{{CNX}}}caption")
        caption = _render(caption_el) if caption_el is not None else ""
        out.append((fig, media.get("alt", "").strip(), caption, image.get("src", "")))
    return out


def pad_square(img: Image.Image, size: int) -> Image.Image:
    """Letterbox onto white instead of cropping: cropping would cut structures off."""
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def ingest_figures(src: SourceCfg, cfg: ImagesCfg, raw_dir: Path, out_dir: Path) -> tuple[list[Figure], list[dict]]:
    """Returns kept figures and the excluded ones with the reason."""
    kept: list[Figure] = []
    dropped: list[dict] = []
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        coll = fetch(src, src.collection, raw_dir, client)
        by_title = chapter_modules(coll, [c.title for c in cfg.chapters])
        for ch in cfg.chapters:
            for idx, mid in enumerate(by_title[ch.title]):
                xml = fetch(src, f"modules/{mid}/index.cnxml", raw_dir, client)
                page_title = _title(ET.fromstring(xml))
                for fig, alt, caption, img_src in _figures_in(xml):
                    fid = fig.get("id", "")
                    title = _title(fig)
                    hit = next((p for p in cfg.exclude_caption_patterns if p in caption or p in title), None)
                    if hit:
                        dropped.append({"id": fid, "module_id": mid, "reason": f"caption/title contains {hit!r}"})
                        continue
                    text = f"{title} {alt}".lower()
                    kind = next((p for p in cfg.exclude_alt_patterns if p.lower() in text), None)
                    if kind:
                        dropped.append({"id": fid, "module_id": mid, "reason": f"not an anatomical drawing ({kind!r})"})
                        continue
                    repo_path = "media/" + img_src.split("media/")[-1]
                    try:
                        data = fetch(src, repo_path, raw_dir, client)
                        original = Image.open(io.BytesIO(data))
                        original.load()
                    except Exception as e:  # missing or unreadable file
                        dropped.append({"id": fid, "module_id": mid, "reason": f"image not usable: {e}"[:200]})
                        continue
                    name = f"{ch.number}.{idx}_{fid}.png"
                    pad_square(original, cfg.resolution).save(out_dir / "img" / name)
                    kept.append(
                        Figure(
                            id=fid,
                            chapter=ch.number,
                            section=f"{ch.number}.{idx}",
                            module_id=mid,
                            title=title,
                            alt=alt,
                            caption=caption,
                            src=raw_url(src, repo_path),
                            file=f"img/{name}",
                            url=page_url(src.book_slug, ch.number, idx, page_title),
                            width=original.width,
                            height=original.height,
                        )
                    )
    return kept, dropped
