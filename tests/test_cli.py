import json

from typer.testing import CliRunner

from medquiz.cli import app
from medquiz.runs import write_jsonl

from .conftest import passing_checks, write_run


def test_rebuild_report_and_release_from_the_command_line(tmp_settings, chunk, question):
    s = tmp_settings
    data = s.path("data")
    write_jsonl(data / "processed" / "chunks.jsonl", [chunk.model_dump()])
    (data / "processed" / "manifest.json").write_text(
        json.dumps({"pages": [{"section": "19.2", "title": "Cardiac Muscle and Electrical Activity"}]})
    )
    write_run(
        s.path("runs"),
        "gen-v1",
        [question],
        {question.id: passing_checks(s)},
        meta={"id": "gen-v1", "kind": "generate", "created_at": "2026-10-18T00:00:00Z"},
    )
    write_jsonl(
        data / "reviews" / "sarthak.jsonl",
        [{"question_id": question.id, "reviewer": "sarthak", "decision": "accept", "reasons": [], "level": "recall",
          "note": "", "seconds": 20.0, "created_at": "2026-10-18T01:00:00Z"}],
    )  # fmt: skip
    runner = CliRunner()

    r = runner.invoke(app, ["db", "rebuild"])
    assert r.exit_code == 0, r.output
    assert "loaded 1 questions and 1 reviews" in r.output

    r = runner.invoke(app, ["report"])
    assert r.exit_code == 0, r.output
    first = (s.path("reports") / "results.md").read_text(encoding="utf-8")
    assert "Rejected: 0.0% (0/1" in first
    # same inputs, byte-identical report: this is what CI relies on
    runner.invoke(app, ["report"])
    assert (s.path("reports") / "results.md").read_text(encoding="utf-8") == first

    r = runner.invoke(app, ["release", "0.1.0", "--run-id", "gen-v1"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["verify-release", "0.1.0"])
    assert r.exit_code == 0 and "all hashes match" in r.output
