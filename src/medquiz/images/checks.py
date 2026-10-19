"""Automatic checks on generated images.

  clip_score   cosine similarity between the image and the subject text (x100).
               Cheap, but CLIP knows little anatomy, so it is reported, never trusted.
  vlm_*        a small vision-language model (qwen2.5-VL 3B) describes what it sees and
               answers yes/no questions: does it show the subject, is there text, is the
               text readable, does it look like a textbook illustration.

The auto verdict rejects an image if the VLM says the subject is not shown or that
the image contains unreadable text. How far to trust that verdict is measured
against the human review, exactly like the question checks."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx
from pydantic import BaseModel


def without_banner(image_path: Path) -> str:
    """The checker sees the picture without the SYNTHETIC banner. Telling the model to
    ignore it did not work: it transcribed the banner as a label."""
    import io

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.crop((0, 0, img.width, int(img.height * 0.95)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


class VLMOut(BaseModel):
    description: str
    shows_subject: bool
    anatomy_problems: str
    contains_text: bool
    text_is_readable: bool
    looks_like_textbook_illustration: bool


VLM_SYSTEM = (
    "You inspect generated medical teaching images. Describe what you actually see, not what the image was "
    "meant to show. Be strict: an image only shows the subject if a student could recognise it. "
    "Reply with JSON only."
)


def vlm_check(client: httpx.Client, model: str, image_path: Path, subject: str) -> tuple[dict, dict]:
    user = (
        f"This image was generated to show: {subject}.\n"
        "description: one or two sentences on what is in the image.\n"
        "shows_subject: true only if the subject is clearly recognisable.\n"
        "anatomy_problems: anything anatomically wrong, missing or extra, or 'none'.\n"
        "contains_text: true if there are letters or labels.\n"
        "text_is_readable: true if all such text is real, readable words; true if there is no text.\n"
        "looks_like_textbook_illustration: true if it looks like a clean diagram from a textbook."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": VLM_SYSTEM},
            {"role": "user", "content": user, "images": [without_banner(image_path)]},
        ],
        "format": VLMOut.model_json_schema(),
        "stream": False,
        "keep_alive": "15m",
        "options": {"temperature": 0, "seed": 3, "num_ctx": 4096, "num_predict": 400},
    }
    t0 = time.perf_counter()
    resp = client.post("/api/chat", json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    out = VLMOut.model_validate_json(data["message"]["content"])
    stats = {
        "seconds": round(time.perf_counter() - t0, 3),
        "tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }
    return out.model_dump(), stats


class ClipScorer:
    def __init__(self, model: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(model).to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained(model)

    def score(self, image_path: Path, text: str) -> float:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        # the synthetic banner is not part of the picture being scored
        img = img.crop((0, 0, img.width, int(img.height * 0.95)))
        inputs = self.proc(text=[text], images=img, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with self.torch.no_grad():
            out = self.model(**inputs)
        a = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        b = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return round(float((a * b).sum()) * 100, 2)

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


class BlindOut(BaseModel):
    description: str
    main_structure: str
    labels: list[str]


BLIND_SYSTEM = (
    "You look at a medical illustration. Say what it actually shows; you are not told what it was meant to "
    "show. Then copy every text label in the image exactly as written, letter by letter, even if it is "
    "misspelled or meaningless. Reply with JSON only."
)

# what the main structure has to be called for the image to count as on-subject
SUBJECT_TERMS = {
    "heart-anterior": ["heart", "cardiac"],
    "heart-chambers": ["heart", "cardiac", "ventric", "atri"],
    "conduction": ["heart", "cardiac", "conduction", "node"],
    "coronary": ["heart", "coronary", "cardiac"],
    "heart-wall": ["heart", "pericard", "myocard", "cardiac"],
    "resp-system": ["respirat", "lung", "airway", "trache"],
    "bronchial-tree": ["bronch", "lung", "airway"],
    "alveoli": ["alveol", "lung", "capillar"],
    "lungs-pleura": ["lung", "pleura", "thora"],
    "larynx": ["larynx", "laryn", "vocal", "throat", "trache"],
    "diaphragm": ["diaphragm", "rib", "thora", "lung"],
    "vessels": ["arter", "vein", "vessel", "capillar"],
}


def vlm_blind(client: httpx.Client, model: str, image_path: Path) -> tuple[dict, dict]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": BLIND_SYSTEM},
            {
                "role": "user",
                "content": "description: one or two sentences. main_structure: the main organ or structure shown. "
                "labels: every text label, copied exactly; an empty list if there are none.",
                "images": [without_banner(image_path)],
            },
        ],
        "format": BlindOut.model_json_schema(),
        "stream": False,
        "keep_alive": "15m",
        "options": {"temperature": 0, "seed": 3, "num_ctx": 4096, "num_predict": 500},
    }
    t0 = time.perf_counter()
    resp = client.post("/api/chat", json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    out = BlindOut.model_validate_json(data["message"]["content"])
    return out.model_dump(), {
        "seconds": round(time.perf_counter() - t0, 3),
        "tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }


def label_words(labels: list[str]) -> list[str]:
    import re

    return [w for lab in labels for w in re.findall(r"[a-z]{3,}", lab.lower())]


def blind_verdict(blind: dict, subject_id: str, vocabulary: set[str]) -> dict:
    """Deterministic tests on the blind answer: is the named structure the target organ,
    and are the label words real terms (they must occur somewhere in the textbook text)?"""
    text = f"{blind['main_structure']} {blind['description']}".lower()
    on_subject = any(t in text for t in SUBJECT_TERMS.get(subject_id, []))
    words = label_words(blind["labels"])
    unknown = [w for w in words if w not in vocabulary]
    garbled = bool(words) and len(unknown) / len(words) >= 0.3
    reasons = (["not_subject"] if not on_subject else []) + (["garbled_text"] if garbled else [])
    return {
        "on_subject": on_subject,
        "n_label_words": len(words),
        "unknown_words": unknown[:15],
        "garbled_labels": garbled,
        "auto": "fail" if reasons else "pass",
        "auto_reasons": reasons,
    }


def auto_verdict(vlm: dict) -> tuple[str, list[str]]:
    reasons = []
    if not vlm["shows_subject"]:
        reasons.append("not_subject")
    if vlm["contains_text"] and not vlm["text_is_readable"]:
        reasons.append("garbled_text")
    return ("fail" if reasons else "pass"), reasons
