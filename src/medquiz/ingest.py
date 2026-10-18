"""Fetch OpenStax CNXML at a pinned commit, turn it into clean text blocks,
pull out the authors' own review questions, and pack blocks into chunks.

Every chunk keeps the paragraph ids it came from, so a citation can point at
the exact paragraph on openstax.org (the page uses the same ids as anchors).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import ChunkCfg, SourceCfg
from .schemas import LETTERS, Block, BookQuestion, Chunk

CNX = "http://cnx.rice.edu/cnxml"
COL = "http://cnx.rice.edu/collxml"
MD = "http://cnx.rice.edu/mdml"

# Sections that hold exercises rather than teaching text.
EXERCISE_SECTION_CLASSES = {
    "review-questions",
    "multiple-choice",
    "critical-thinking",
    "interactive-exercise",
    "summary",
    "key-terms",
}
SKIP_INLINE = {"figure", "media", "image", "table", "exercise", "footnote", "label", "title"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# ---------------------------------------------------------------- fetching


def raw_url(src: SourceCfg, path: str) -> str:
    return f"https://raw.githubusercontent.com/{src.repo}/{src.commit}/{path}"


def fetch(src: SourceCfg, path: str, raw_dir: Path, client: httpx.Client) -> bytes:
    """Download one file from the source repo, caching it under raw_dir/<commit>/."""
    dest = raw_dir / src.commit / path
    if dest.exists():
        return dest.read_bytes()
    resp = client.get(raw_url(src, path), follow_redirects=True, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return resp.content


def chapter_modules(collection_xml: bytes, titles: list[str]) -> dict[str, list[str]]:
    """Map chapter title -> module ids in book order (first one is the intro page)."""
    root = ET.fromstring(collection_xml)
    found: dict[str, list[str]] = {}
    for sub in root.iter(f"{{{COL}}}subcollection"):
        title_el = sub.find(f"{{{MD}}}title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if title in titles:
            content = sub.find(f"{{{COL}}}content")
            found[title] = [m.get("document") for m in content.findall(f"{{{COL}}}module")]
    missing = set(titles) - set(found)
    if missing:
        raise ValueError(f"chapters not found in collection: {sorted(missing)}")
    return found


# ------------------------------------------------------------------ text


def _render(el: ET.Element) -> str:
    out: list[str] = []

    def rec(e: ET.Element) -> None:
        tag = _local(e.tag)
        if tag in SKIP_INLINE and e is not el:
            return
        if tag == "link" and not "".join(e.itertext()).strip():
            target = e.get("target-id", "")
            out.append("[Figure]" if "fig" in target else "[Table]" if "table" in target else "")
            return
        if tag == "newline":
            out.append(" ")
            return
        if tag == "item":
            out.append(" ")
        if e.text:
            out.append(e.text)
        for child in e:
            rec(child)
            if child.tail:
                out.append(child.tail)
        if tag == "item":
            out.append(";")

    rec(el)
    return clean_text("".join(out))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    # "(see [Figure])", "([Figure])", "[Figure] and [Figure]" -> drop or soften
    text = re.sub(r"\s*\((?:see |and |, )*(?:\[(?:Figure|Table)\](?:,? (?:and )?)?)+\)", "", text)
    text = re.sub(
        r"\[(Figure|Table)\][a-d]?(?: and \[(?:Figure|Table)\][a-d]?)*",
        lambda m: "the " + m.group(1).lower(),
        text,
    )
    text = re.sub(r"(^|[.!?]\s+)the (figure|table)", lambda m: m.group(1) + "The " + m.group(2), text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r";\s*$", "", text.strip())
    return text.strip()


def _title(el: ET.Element) -> str:
    t = el.find(f"{{{CNX}}}title")
    return clean_text("".join(t.itertext())) if t is not None else ""


# ----------------------------------------------------------------- parse


@dataclass
class ParsedModule:
    module_id: str
    chapter: int
    index: int  # 0 = chapter intro
    title: str
    objectives: list[str]
    blocks: list[Block] = field(default_factory=list)
    questions: list[BookQuestion] = field(default_factory=list)

    @property
    def section(self) -> str:
        return f"{self.chapter}.{self.index}"


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", title.lower())
    return re.sub(r"[\s-]+", "-", s).strip("-")


def page_url(book_slug: str, chapter: int, index: int, title: str) -> str:
    page = f"{chapter}-introduction" if index == 0 else f"{chapter}-{index}-{slugify(title)}"
    return f"https://openstax.org/books/{book_slug}/pages/{page}"


def parse_module(xml: bytes, module_id: str, chapter: int, index: int, keep_notes: set[str]) -> ParsedModule:
    root = ET.fromstring(xml)
    title = _title(root)
    objectives = []
    abstract = root.find(f".//{{{MD}}}abstract")
    if abstract is not None:
        objectives = [_render(i) for i in abstract.iter(f"{{{CNX}}}item")]
    mod = ParsedModule(module_id, chapter, index, title, objectives)
    content = root.find(f"{{{CNX}}}content")
    if content is not None:
        _walk(content, [], mod, keep_notes)
    return mod


def _walk(el: ET.Element, heading: list[str], mod: ParsedModule, keep_notes: set[str]) -> None:
    for child in el:
        tag = _local(child.tag)
        cid = child.get("id", "")
        if tag == "section":
            classes = set(child.get("class", "").split())
            if classes & EXERCISE_SECTION_CLASSES:
                if "multiple-choice" in classes:
                    mod.questions.extend(_parse_mcqs(child, mod))
                continue
            title = _title(child)
            if title.lower() in {"references", "chapter review"}:
                continue
            _walk(child, heading + [title], mod, keep_notes)
        elif tag == "para":
            text = _render(child)
            if len(text.split()) >= 5:
                mod.blocks.append(Block(id=cid, module_id=mod.module_id, heading=heading, kind="para", text=text))
        elif tag == "list":
            items = [_render(i) for i in child.findall(f"{{{CNX}}}item")]
            items = [i for i in items if i]
            if items:
                text = "\n".join(f"- {i}" for i in items)
                mod.blocks.append(Block(id=cid, module_id=mod.module_id, heading=heading, kind="list", text=text))
        elif tag == "note":
            classes = set(child.get("class", "").split())
            if not classes & keep_notes:
                continue
            parts = [_render(p) for p in child if _local(p.tag) in ("para", "list")]
            body = " ".join(p for p in parts if p)
            if len(body.split()) >= 20:
                label = _title(child)
                text = f"{label}. {body}" if label else body
                mod.blocks.append(Block(id=cid, module_id=mod.module_id, heading=heading, kind="note", text=text))


def _parse_mcqs(section: ET.Element, mod: ParsedModule) -> list[BookQuestion]:
    out = []
    for ex in section.iter(f"{{{CNX}}}exercise"):
        problem = ex.find(f"{{{CNX}}}problem")
        solution = ex.find(f"{{{CNX}}}solution")
        if problem is None or solution is None:
            continue
        stem_el = problem.find(f"{{{CNX}}}para")
        list_el = problem.find(f".//{{{CNX}}}list")
        if stem_el is None or list_el is None:
            continue
        stem = _render(stem_el)
        options = [_render(i) for i in list_el.findall(f"{{{CNX}}}item")]
        key = clean_text("".join(solution.itertext())).strip().upper()[:1]
        if len(options) != 4 or key not in LETTERS or not stem:
            continue
        out.append(
            BookQuestion(
                id=f"book-{mod.module_id}-{ex.get('id', len(out))}",
                chapter=mod.chapter,
                section=mod.section,
                module_id=mod.module_id,
                stem=stem,
                options=options,
                answer_index=LETTERS.index(key),
            )
        )
    return out


# ----------------------------------------------------------------- chunk


# a sentence may end inside a closing quote or bracket: ... the Greek term, "kardia." Cardiology ...
_SENT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][”\"’')\]]))\s+(?=[A-Z(“\"])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _units(block: Block, cfg: ChunkCfg) -> list[tuple[Block, str]]:
    """A block longer than max_words is cut at sentence boundaries."""
    if len(block.text.split()) <= cfg.max_words:
        return [(block, block.text)]
    pieces, cur = [], []
    for s in split_sentences(block.text):
        cur.append(s)
        if len(" ".join(cur).split()) >= cfg.target_words:
            pieces.append(" ".join(cur))
            cur = []
    if cur:
        pieces.append(" ".join(cur))
    return [(block, p) for p in pieces]


def chunk_module(mod: ParsedModule, cfg: ChunkCfg, book_slug: str) -> list[Chunk]:
    """Greedy packing of whole paragraphs. A chunk never crosses a top-level
    section of the page, so its heading is always accurate."""
    url = page_url(book_slug, mod.chapter, mod.index, mod.title)
    groups: list[list[tuple[Block, str]]] = []
    last_key = None
    for b in mod.blocks:
        key = tuple(b.heading[:1])
        if key != last_key:
            groups.append([])
            last_key = key
        groups[-1].extend(_units(b, cfg))
    # A short lead-in before the first heading joins the section that follows it.
    for i in range(len(groups) - 2, -1, -1):
        if sum(len(t.split()) for _, t in groups[i]) < cfg.min_words:
            groups[i + 1] = groups[i] + groups[i + 1]
            del groups[i]

    packed: list[list[tuple[Block, str]]] = []
    for units in groups:
        chunks: list[list[tuple[Block, str]]] = []
        cur: list[tuple[Block, str]] = []
        n = 0
        for u in units:
            w = len(u[1].split())
            if cur and (n >= cfg.target_words or (n + w > cfg.max_words and n >= cfg.min_words)):
                chunks.append(cur)
                cur, n = [], 0
            cur.append(u)
            n += w
        if cur:
            prev_n = sum(len(t.split()) for _, t in chunks[-1]) if chunks else 0
            if chunks and n < cfg.min_words and prev_n + n <= int(cfg.max_words * 1.3):
                chunks[-1].extend(cur)
            else:
                chunks.append(cur)
        packed.extend(chunks)

    out = []
    for i, units in enumerate(packed):
        first = units[0][0]
        text = "\n\n".join(t for _, t in units)
        heading = " > ".join([mod.title, *first.heading]) if first.heading else mod.title
        out.append(
            Chunk(
                id=f"{mod.section}-{i:03d}",
                chapter=mod.chapter,
                section=mod.section,
                section_title=mod.title,
                heading=heading,
                module_id=mod.module_id,
                para_ids=list(dict.fromkeys(b.id for b, _ in units)),
                text=text,
                n_words=len(text.split()),
                url=url,
            )
        )
    return out


def ingest(src: SourceCfg, cfg: ChunkCfg, raw_dir: Path) -> tuple[list[ParsedModule], list[Chunk], list[BookQuestion]]:
    keep = set(src.keep_note_classes)
    with httpx.Client() as client:
        coll = fetch(src, src.collection, raw_dir, client)
        by_title = chapter_modules(coll, [c.title for c in src.chapters])
        modules: list[ParsedModule] = []
        for ch in src.chapters:
            for idx, mid in enumerate(by_title[ch.title]):
                xml = fetch(src, f"modules/{mid}/index.cnxml", raw_dir, client)
                modules.append(parse_module(xml, mid, ch.number, idx, keep))
    chunks = [c for m in modules for c in chunk_module(m, cfg, src.book_slug)]
    questions = [q for m in modules for q in m.questions]
    return modules, chunks, questions
