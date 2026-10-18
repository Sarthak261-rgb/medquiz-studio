"""Prompt files live in prompts/<name>.md with a "## system" and a "## user" part.
The file name is the version; the hash goes into every record that used it."""

from __future__ import annotations

from string import Template

from .config import load_prompt
from .schemas import LETTERS


def render(name: str, **values: str) -> tuple[str, str, str]:
    text, digest = load_prompt(name)
    head, _, user = text.partition("## user")
    system = head.replace("## system", "", 1).strip()
    return (
        Template(system).safe_substitute(values),
        Template(user.strip()).substitute(values),
        digest,
    )


def format_options(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
