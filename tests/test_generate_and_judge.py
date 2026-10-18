import json

import httpx
import pytest

from medquiz.checks.judge import audit, blind_solve, closed_book
from medquiz.checks.nli import distractor_statements
from medquiz.generate import generate_question, shuffle_options, target_for
from medquiz.llm import LLMError, OllamaClient
from medquiz.mutate import _replace_key
from medquiz.schemas import MCQDraft

from .conftest import FakeLLM

DRAFT = {
    "stem": "Which structure is the normal pacemaker of the heart?",
    "options": ["sinoatrial node", "atrioventricular node", "bundle of His", "Purkinje fibers"],
    "answer_index": 0,
    "explanation": "The SA node initiates the sinus rhythm.",
    "evidence_quote": "The sinoatrial (SA) node is the normal pacemaker of the heart.",
    "answer_statement": "The sinoatrial node is the normal pacemaker of the heart.",
    "bloom_level": "remember",
}


def test_generation_shuffles_but_keeps_the_key(settings, chunk):
    llm = FakeLLM({"MCQDraft": [DRAFT, DRAFT]})
    q = generate_question(llm, settings.generator, chunk, "run-a", 0, "recall")
    assert q.answer == "sinoatrial node"
    assert sorted(q.options) == sorted(DRAFT["options"])
    assert q.gen.original_answer_index == 0
    assert q.gen.prompt_tokens == 100 and q.gen.prompt_hash
    # passage and target type reach the prompt
    _, system, user = llm.calls[0]
    assert chunk.text in user and "recall" in user and "according to the passage" in system
    # same run, chunk and k -> same id and same shuffle
    q2 = generate_question(llm, settings.generator, chunk, "run-a", 0, "recall")
    assert (q2.id, q2.options) == (q.id, q.options)


def test_shuffle_is_seeded_and_spreads_the_key():
    positions = {shuffle_options(list("abcd"), 0, f"q{i}")[1] for i in range(40)}
    assert positions == {0, 1, 2, 3}
    assert shuffle_options(list("abcd"), 2, "x") == shuffle_options(list("abcd"), 2, "x")
    assert [target_for(i) for i in range(4)] == ["recall", "reasoning", "recall", "reasoning"]


def test_ollama_client_retries_invalid_json_and_sums_tokens():
    replies = iter(
        [
            {"message": {"content": '{"stem": "too short"}'}, "prompt_eval_count": 500, "eval_count": 20},
            {"message": {"content": json.dumps(DRAFT)}, "prompt_eval_count": 500, "eval_count": 150},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"]["type"] == "object" and body["stream"] is False
        return httpx.Response(200, json=next(replies))

    client = OllamaClient()
    client.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    draft, stats = client.chat_json("m", "sys", "user", MCQDraft, seed=1)
    assert draft.options[0] == "sinoatrial node"
    assert stats.attempts == 2 and stats.prompt_tokens == 1000 and stats.completion_tokens == 170


def test_ollama_client_gives_up():
    client = OllamaClient()
    client.http = httpx.Client(
        base_url="http://x",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"message": {"content": "not json"}})),
    )
    with pytest.raises(LLMError):
        client.chat_json("m", "s", "u", MCQDraft, max_attempts=2)


def _verdicts(key="B", also=None, low=()):
    rows = []
    for L in "ABCD":
        verdict = "supported" if L == key or L == also else "contradicted"
        plaus = "n/a" if L == key else ("low" if L in low else "medium")
        rows.append({"option": L, "evidence": "quote", "verdict": verdict, "plausibility": plaus})
    return {"options": rows, "cognitive_level": "recall", "stem_clear": True}


def test_audit_derivations(settings, question, chunk):
    judge = settings.judge
    good = {r.name: r for r in audit(FakeLLM({"AuditOut": [_verdicts()]}), judge, question, chunk.text)}
    assert good["judge_answer_grounded"].passed and good["judge_single_correct"].passed
    assert good["judge_distractor_quality"].passed and good["judge_level"].detail["level"] == "recall"
    assert good["judge_answer_grounded"].tokens == 150 and good["judge_single_correct"].tokens == 0

    two = {r.name: r for r in audit(FakeLLM({"AuditOut": [_verdicts(also="D")]}), judge, question, chunk.text)}
    assert two["judge_single_correct"].passed is False and two["judge_single_correct"].detail["also_supported"] == ["D"]

    weak = {r.name: r for r in audit(FakeLLM({"AuditOut": [_verdicts(low="AC")]}), judge, question, chunk.text)}
    assert weak["judge_distractor_quality"].passed is False

    ungrounded = {r.name: r for r in audit(FakeLLM({"AuditOut": [_verdicts(key="A")]}), judge, question, chunk.text)}
    assert ungrounded["judge_answer_grounded"].passed is False


def test_blind_and_closed_book_solve(settings, question, chunk):
    llm = FakeLLM(
        {
            "SolveOut": [
                {"reason": "r", "answer": "B", "confidence": 5},
                {"reason": "r", "answer": "A", "confidence": 2},
                {"reason": "r", "answer": "B", "confidence": 4},
            ]
        }
    )
    assert blind_solve(llm, settings.judge, question, chunk.text).passed is True
    wrong = blind_solve(llm, settings.judge, question, chunk.text)
    assert wrong.passed is False and wrong.detail["picked_text"] == "atrioventricular node"
    cb = closed_book(llm, settings.judge, question)
    assert cb.passed is True
    assert "Passage" not in llm.calls[-1][2]


def test_distractor_statements_swap_the_key(question):
    stmts = distractor_statements(question)
    assert stmts[0] == "The atrioventricular node is the normal pacemaker of the heart."
    assert len(stmts) == 3
    q = question.model_copy(update={"answer_statement": "It sets the rhythm."})
    assert distractor_statements(q) == [None, None, None]
    assert _replace_key("The SA node fires.", "SA node", "AV node") == "The AV node fires."
    assert _replace_key("Nothing here.", "SA node", "AV node") == ""


def test_nli_lexical_guard():
    from medquiz.checks.nli import mentions

    span = "Cardiology is the study of the heart, and cardiologists are the physicians who treat it."
    assert mentions(span, "cardiology")
    assert not mentions(span, "hematology")
    assert mentions(span, "the")  # nothing to check, so it does not block


def test_self_judge_uses_its_own_names(settings, question, chunk):
    from medquiz.checks.judge import self_judge_checks

    llm = FakeLLM({"SolveOut": [{"reason": "r", "answer": "B", "confidence": 5}], "AuditOut": [_verdicts()]})
    names = [r.name for r in self_judge_checks(llm, settings.self_judge, question, chunk.text)]
    assert names == [
        "self_blind_solve",
        "self_answer_grounded",
        "self_single_correct",
        "self_distractor_quality",
        "self_stem_clear",
        "self_level",
    ]
    assert llm.calls[0][0] == settings.self_judge.model


def test_options_only_hides_stem_and_passage(settings, question):
    from medquiz.checks.judge import options_only

    llm = FakeLLM({"SolveOut": [{"reason": "r", "answer": "B", "confidence": 2}]})
    r = options_only(llm, settings.judge, question)
    assert r.name == "options_only_solve" and r.passed is True
    _, system, user = llm.calls[0]
    assert question.stem not in user and "sinoatrial node" in user
