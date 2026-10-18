from medquiz.checks import auto_verdict
from medquiz.checks.rules import (
    evidence_in_source,
    length_cue,
    no_all_none_of_above,
    options_distinct,
    statement_consistent,
    stem_leak,
    text_reference,
)
from medquiz.schemas import CheckResult, FailureType, Question


def test_clean_question_passes_every_rule(question, chunk):
    for r in [
        options_distinct(question),
        no_all_none_of_above(question),
        stem_leak(question),
        text_reference(question),
        length_cue(question, 1.6),
        evidence_in_source(question, chunk, 90),
        statement_consistent(question),
    ]:
        assert r.passed is True, r


def test_near_duplicate_options_fail(question):
    q = question.model_copy(update={"options": ["sinoatrial node", "the sinoatrial node", "AV node", "bundle"]})
    assert options_distinct(q).passed is False


def test_all_and_none_of_the_above(question):
    for bad in ["All of the above", "none of these", "Both A and B", "A and C"]:
        q = question.model_copy(update={"options": ["a node", "sinoatrial node", bad, "fibers"]})
        assert no_all_none_of_above(q).passed is False, bad


def test_key_in_stem_is_a_leak(question):
    q = question.model_copy(update={"stem": "The sinoatrial node is which part of the heart's conduction system?"})
    assert stem_leak(q).passed is False


def test_text_reference(question):
    for stem in [
        "According to the passage, what sets the pace?",
        "What term is used in the passage for X?",
        "Based on the information provided, which is true?",
    ]:
        assert text_reference(question.model_copy(update={"stem": stem})).passed is False, stem


def test_length_cue(question):
    q = question.model_copy(
        update={
            "options": ["AV node", "the sinoatrial node located in the right atrium wall", "His bundle", "Purkinje"]
        }
    )
    r = length_cue(q, 1.6)
    assert r.passed is False and r.score > 1.6


def test_evidence_quote_must_come_from_the_chunk(question, chunk):
    fake = question.model_copy(update={"evidence_quote": "The SA node is located in the left ventricle near the apex."})
    assert evidence_in_source(fake, chunk, 90).passed is False
    # small punctuation/case differences are tolerated
    close = question.model_copy(
        update={"evidence_quote": "the sinoatrial (SA) node is the normal pacemaker of the heart"}
    )
    assert evidence_in_source(close, chunk, 90).passed is True


def test_statement_must_contain_the_key(question):
    q = question.model_copy(update={"answer_statement": "The AV node delays the impulse."})
    assert statement_consistent(q).passed is False


def _c(name, passed, detail=None):
    return CheckResult(name=name, checker="rule", passed=passed, detail=detail or {})


def test_auto_verdict(settings):
    hard = settings.checks.hard
    ok = [_c(n, True) for n in hard]
    assert auto_verdict(ok, settings.checks).status == "pass"

    failing = [_c(n, n != "judge_single_correct") for n in hard]
    v = auto_verdict(failing, settings.checks)
    assert v.status == "fail" and v.primary == FailureType.multiple_correct

    errored = [_c(n, None, {"error": "timeout"}) if n == "judge_blind_solve" else _c(n, True) for n in hard]
    assert auto_verdict(errored, settings.checks).status == "incomplete"

    # a check that does not apply is ignored, not counted as missing
    no_dup = [c for c in ok if c.name != "duplicate"]
    assert auto_verdict(no_dup, settings.checks, set(hard) - {"duplicate"}).status == "pass"

    # soft failures never block
    soft = ok + [_c("length_cue", False)]
    v = auto_verdict(soft, settings.checks)
    assert v.status == "pass" and v.failed_soft == ["length_cue"]


def test_negative_stem():
    from medquiz.checks.rules import negative_stem

    def q(stem):
        return Question(id="x", run_id="r", chunk_id="c", stem=stem, options=list("abcd"), answer_index=0)

    assert negative_stem(q("Which of the following is NOT a heart valve?")).passed is False
    assert negative_stem(q("Which of these is not part of the conduction system?")).passed is False
    assert negative_stem(q("All of the following are true EXCEPT:")).passed is False
    # a negation inside a scenario is fine
    assert negative_stem(q("If the mitral valve does not close fully, where does blood flow?")).passed is True


def test_absolute_terms(question):
    from medquiz.checks.rules import absolute_terms

    q = question.model_copy(update={"options": ["It never fires", "sinoatrial node", "It always delays", "fibers"]})
    r = absolute_terms(q)
    assert r.passed is False and r.detail["flagged"] == [0, 2]
    assert absolute_terms(question).passed is True


def test_numeric_grounding(question, chunk):
    from medquiz.checks.rules import numeric_grounding

    ok = question.model_copy(update={"options": ["50 ms", "100 ms", "150 ms", "200 ms"], "answer_index": 1})
    assert numeric_grounding(ok, chunk).passed is True
    bad = ok.model_copy(update={"answer_index": 3})
    r = numeric_grounding(bad, chunk)
    assert r.passed is False and r.detail["missing"] == ["200"]
    assert numeric_grounding(question, chunk).passed is None  # no number in the key


def test_key_terms_in_source(question, chunk):
    from medquiz.checks.rules import key_terms_in_source

    assert key_terms_in_source(question, chunk).passed is True
    off = question.model_copy(
        update={"options": ["vagus nerve", "baroreceptor reflex arc", "c", "d"], "answer_index": 1}
    )
    assert key_terms_in_source(off, chunk).passed is False
