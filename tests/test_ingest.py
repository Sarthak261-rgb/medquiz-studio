from medquiz.config import ChunkCfg
from medquiz.ingest import chapter_modules, chunk_module, clean_text, page_url, parse_module, slugify

from .conftest import FIXTURES

KEEP = {"disorders", "diseases", "everyday"}


def parse():
    return parse_module((FIXTURES / "mini.cnxml").read_bytes(), "m00001", 19, 1, KEEP)


def test_parse_blocks_and_headings():
    mod = parse()
    assert mod.title == "Heart Anatomy"
    assert mod.objectives == ["Describe the location of the heart", "Name the chambers"]
    ids = [b.id for b in mod.blocks]
    assert ids == ["p-lead", "p1", "n2", "p2", "l1"]
    by_id = {b.id: b for b in mod.blocks}
    assert by_id["p-lead"].heading == []
    assert by_id["p1"].heading == ["Location of the Heart"]
    assert by_id["n2"].kind == "note" and by_id["n2"].text.startswith("Disorders of the Heart: Pericarditis.")
    assert by_id["l1"].text.splitlines() == ["- right atrium", "- right ventricle", "- left atrium", "- left ventricle"]


def test_figures_references_and_video_boxes_are_dropped():
    text = " ".join(b.text for b in parse().blocks)
    assert "Caption text" not in text
    assert "video" not in text  # interactive box skipped
    assert "Someone, A." not in text  # References section skipped
    assert "[Figure]" not in text
    p1 = next(b for b in parse().blocks if b.id == "p1")
    assert "The figure shows the position of the heart." in p1.text
    assert "(see" not in p1.text


def test_review_questions_are_extracted_with_keys():
    qs = parse().questions
    assert len(qs) == 1  # the two-option exercise is skipped
    q = qs[0]
    assert q.options == ["mitral", "tricuspid", "pulmonary", "aortic"]
    assert q.answer_index == 0
    assert q.section == "19.1"


def test_chunking_keeps_every_word_once():
    mod = parse()
    chunks = chunk_module(mod, ChunkCfg(target_words=30, max_words=60, min_words=10), "book")
    assert sum(c.n_words for c in chunks) == sum(len(b.text.split()) for b in mod.blocks)
    assert len({c.id for c in chunks}) == len(chunks)
    # a chunk never mixes two top-level sections of the page
    for c in chunks:
        assert c.heading.startswith("Heart Anatomy")
    assert all(c.url == "https://openstax.org/books/book/pages/19-1-heart-anatomy" for c in chunks)


def test_short_lead_in_joins_next_section():
    mod = parse()
    chunks = chunk_module(mod, ChunkCfg(target_words=200, max_words=300, min_words=40), "book")
    first = chunks[0]
    assert first.para_ids[:2] == ["p-lead", "p1"]


def test_long_paragraph_is_split_on_sentences():
    mod = parse()
    long = mod.blocks[1].model_copy(update={"text": "Sentence number one is here. " * 40})
    mod.blocks = [long]
    chunks = chunk_module(mod, ChunkCfg(target_words=50, max_words=60, min_words=10), "book")
    assert len(chunks) > 1
    assert all(c.n_words <= 60 for c in chunks)


def test_clean_text_and_urls():
    assert clean_text("shown in [Figure]b , and ([Figure]) here .") == "shown in the figure, and here."
    assert clean_text("[Figure] shows it") == "The figure shows it"
    assert (
        slugify("Organs and Structures of the Respiratory System") == "organs-and-structures-of-the-respiratory-system"
    )
    assert page_url("b", 22, 0, "Introduction").endswith("/22-introduction")


def test_chapter_modules_from_collection():
    xml = b"""<col:collection xmlns:col="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml">
      <col:content><col:subcollection><md:title>Unit</md:title><col:content>
        <col:subcollection><md:title>The Respiratory System</md:title><col:content>
          <col:module document="m1"/><col:module document="m2"/></col:content></col:subcollection>
      </col:content></col:subcollection></col:content></col:collection>"""
    assert chapter_modules(xml, ["The Respiratory System"]) == {"The Respiratory System": ["m1", "m2"]}
