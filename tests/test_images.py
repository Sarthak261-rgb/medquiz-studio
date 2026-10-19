import json

import pytest
from PIL import Image

from medquiz.images.checks import auto_verdict
from medquiz.images.figures import Figure, pad_square
from medquiz.images.generate import save_with_metadata, stamp
from medquiz.images.lora import caption_for
from medquiz.images.metrics import images_summary
from medquiz.images.review import ImageReviewIn, add_image_review, latest_image_reviews, queue

STAMP = "SYNTHETIC EDUCATIONAL IMAGE - NOT A REAL PATIENT"


def test_every_image_is_marked_synthetic_twice(workdir):
    img = stamp(Image.new("RGB", (512, 512), "white"), STAMP)
    # the banner is drawn into the pixels at the bottom
    assert img.getpixel((256, 508)) != (255, 255, 255)
    assert img.getpixel((256, 100)) == (255, 255, 255)
    path = workdir / "x.jpg"
    save_with_metadata(img, path, "prompt", 7, "sd15", STAMP)
    exif = Image.open(path).getexif()
    assert exif[0x010E].startswith(STAMP) and "seed 7" in exif[0x010E]
    assert exif[0x0131] == "MedQuiz Studio"


def test_pad_square_keeps_the_whole_figure():
    wide = Image.new("RGB", (1000, 400), "red")
    out = pad_square(wide, 512)
    assert out.size == (512, 512)
    assert out.getpixel((256, 10)) == (255, 255, 255)  # letterboxed, not cropped
    assert out.getpixel((256, 256))[0] > 200


def test_lora_caption():
    f = Figure(id="f", chapter=19, section="19.1", module_id="m", title="Heart Valves", alt="This diagram shows the valves.",
               caption="", src="", file="", url="", width=1, height=1)  # fmt: skip
    assert caption_for(f, "osxillus") == "osxillus, medical textbook illustration of heart valves, the valves"


def test_image_auto_verdict():
    ok = {"shows_subject": True, "contains_text": True, "text_is_readable": True}
    assert auto_verdict(ok) == ("pass", [])
    assert auto_verdict({**ok, "text_is_readable": False}) == ("fail", ["garbled_text"])
    assert auto_verdict({**ok, "shows_subject": False}) == ("fail", ["not_subject"])


def test_image_review_rules_and_queue(workdir):
    with pytest.raises(ValueError):
        ImageReviewIn(reviewer="s", decision="reject")
    with pytest.raises(ValueError):
        ImageReviewIn(reviewer="s", decision="reject", reasons=["made_up"])
    add_image_review(
        workdir, "lora-a-1", ImageReviewIn(reviewer="sarthak", decision="reject", reasons=["garbled_text"])
    )
    add_image_review(workdir, "base-a-1", ImageReviewIn(reviewer="sarthak", decision="accept"))
    got = latest_image_reviews(workdir, "sarthak")
    assert set(got) == {"lora-a-1", "base-a-1"}
    images = [{"id": i} for i in ("base-a-1", "lora-a-1", "base-b-1", "lora-b-1")]
    q = queue(images, set(got))
    assert [r["id"] for r in q] == [r["id"] for r in queue(images, set(got))]  # stable
    assert {r["id"] for r in q} == {"base-b-1", "lora-b-1"}


def test_images_summary(tmp_settings):
    s = tmp_settings
    run = s.path("runs") / "images-v1"
    run.mkdir(parents=True)
    rows = [
        {"id": f"{arm}-heart-{seed}", "arm": arm, "subject_id": "heart", "subject": "the heart", "seed": seed,
         "file": "x.jpg", "seconds": 7.0, "gpu_joules": 360.0}
        for arm in ("base", "lora") for seed in (1, 2)
    ]  # fmt: skip
    (run / "images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    vlm = {"shows_subject": True, "contains_text": False, "text_is_readable": True,
           "looks_like_textbook_illustration": True}  # fmt: skip
    (run / "image_checks.jsonl").write_text(
        "".join(json.dumps({"id": r["id"], "clip_score": 30.0, "vlm": vlm, "auto": "pass"}) + "\n" for r in rows)
    )
    log = s.path("data") / "reviews"
    add_image_review(
        log, "base-heart-1", ImageReviewIn(reviewer="sarthak", decision="reject", reasons=["wrong_anatomy"])
    )
    add_image_review(log, "lora-heart-1", ImageReviewIn(reviewer="sarthak", decision="accept"))
    m = images_summary(s)
    assert m["reviewer"] == "sarthak"
    assert m["arms"]["base"]["human"]["rejection"]["k"] == 1
    assert m["arms"]["lora"]["human"]["rejection"]["k"] == 0
    assert m["arms"]["lora"]["gpu_wh_per_image"] == 0.1
    assert m["arms"]["lora"]["gpu_wh_per_accepted"] == 0.2
