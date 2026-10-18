"""Static figures for the README, drawn from metrics.json. Light surface only;
GitHub shows PNGs as they are."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .schemas import FAILURE_LABELS, FailureType  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # validated pair: CVD dE 24.7, normal dE 33.6


def _style(ax, xmax: float) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, xmax)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def _label(ftype: str) -> str:
    try:
        return FAILURE_LABELS[FailureType(ftype)]
    except ValueError:
        return ftype


def failure_breakdown(m: dict, out: Path) -> Path | None:
    human = {k: v["rate"] for k, v in m["human"]["primary_reason"].items()}
    auto = {k: v["rate"] for k, v in m["auto"]["primary_failure"].items()}
    keys = sorted(set(human) | set(auto), key=lambda k: max(human.get(k, 0), auto.get(k, 0)))
    if not keys:
        return None
    series = [("Automatic checks", auto)]
    if human:
        series.insert(0, (f"Human review (n={m['human']['n_reviewed']})", human))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(keys) + 1.4), dpi=150, facecolor=SURFACE)
    h = 0.34 if len(series) == 2 else 0.5
    xmax = max(max(d.values(), default=0) for _, d in series) * 1.25 or 1
    for i, (name, d) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * (h + 0.04)
        ys = [y - offset for y in range(len(keys))]
        vals = [d.get(k, 0) for k in keys]
        ax.barh(ys, vals, height=h, color=SERIES[i], label=name)
        for y, v in zip(ys, vals, strict=True):
            if v:
                ax.text(v + xmax * 0.01, y, f"{100 * v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(keys)), [_label(k) for k in keys], color=INK, fontsize=9)
    _style(ax, xmax)
    ax.set_title("Primary failure, share of generated questions", loc="left", color=INK, fontsize=11, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    path = out / "failure_breakdown.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def stress_test(m: dict, out: Path) -> Path | None:
    st = m.get("stress_test")
    if not st:
        return None
    names = [k for k in st if k != "reorder"]
    vals = [st[k]["auto_reject"]["rate"] or 0 for k in names]
    order = sorted(range(len(names)), key=lambda i: vals[i])
    names, vals = [names[i] for i in order], [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.3), dpi=150, facecolor=SURFACE)
    ax.barh(range(len(names)), vals, height=0.5, color=SERIES[0])
    for y, v in enumerate(vals):
        ax.text(v + 0.012, y, f"{100 * v:.0f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], color=INK, fontsize=9)
    _style(ax, 1.12)
    n = st[names[0]]["n"]
    ax.set_title(
        f"Planted defects caught by the automatic checks (n={n} per row)", loc="left", color=INK, fontsize=11, pad=10
    )
    fig.tight_layout()
    path = out / "stress_test.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_figures(m: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [p for p in (failure_breakdown(m, out), stress_test(m, out)) if p]
