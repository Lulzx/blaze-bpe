#!/usr/bin/env python3
"""Regenerate the README benchmark charts, adding a gigatoken series.

All non-gigatoken data points are the existing plugged-in, best-of numbers from
the published charts (kept identical for consistency). Only the gigatoken series
is freshly measured (see scripts/bench_gigatoken.sh) and filled in below.

Usage: python3 scripts/make_charts.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

BG = "#fbfbf6"
C = {
    "blaze_par": "#2f76d1",
    "blaze_1t":  "#a9c9ef",
    "gigatoken": "#cc3333",
    "quicktok":  "#4b3f9e",
    "rs_bpe":    "#28b487",
    "tiktoken":  "#f4a81d",
    "tokenizers": "#1f8a2e",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.size": 15,
    "axes.edgecolor": "#cccccc",
})

# ---- FRESH same-session FIRST-RUN numbers (tokens/s), measured on AC --------
# First-run methodology: every measurement is a fresh process, so cache
# population is included in the timed pass (no warmup). This is the honest
# number for a one-shot bulk tokenization and is apples-to-apples with
# gigatoken's fresh-process `bench`. Best of 5 fresh runs (file warm in the OS
# page cache, tokenizer cache cold each time), gigatoken and blaze interleaved
# in one thermal session. Single-thread rivals kept at their documented values
# (single-thread is reproducible; on this repetitive 1 GB corpus first-run and
# warm converge within the first chunk). Only the 1 GB regime is charted.
GT_1GB_CL = 1277.8e6  # 1 GB shakespeare x900, cl100k, first-run best-of-5
GT_1GB_O2 = 1084.6e6  # 1 GB shakespeare x900, o200k, first-run best-of-5
# ---------------------------------------------------------------------------

TOK_CL = 271_646_100
TOK_O2 = 267_845_400


def m_lin(v):  # bar-chart label: always millions, comma, 1 decimal
    return f"{v/1e6:,.1f}M"


def m_log(v):  # line-chart point label
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e7:
        return f"{v/1e6:.0f}M"
    return f"{v/1e6:.1f}M"


def bar_chart(path, title, footer, total_tokens, rows):
    rows = sorted(rows, key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(14.1, 5.6), dpi=150)
    y = list(range(len(rows)))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    xmax = max(vals)
    for yi, v in zip(y, vals):
        secs = total_tokens / v
        ax.text(v + xmax * 0.014, yi,
                f"{m_lin(v)} tok/s   ({secs:.1f} s for the whole GB)",
                va="center", ha="left", fontweight="bold", fontsize=13.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#555555")
    ax.set_xlim(0, xmax * 1.9)
    ax.set_xlabel("Tokens per second (linear scale)")
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: "0" if x == 0 else (f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M")))
    ax.set_title(title, fontweight="bold", fontsize=16.5, pad=14)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#e3e3dc", zorder=0)
    fig.text(0.011, 0.02, footer, fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=0.16, right=0.985, top=0.88, bottom=0.17)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def line_chart(path, title, footer, xticklabels, xlabel, series, ymin, ymax):
    fig, ax = plt.subplots(figsize=(13.2, 7.2), dpi=150)
    x = list(range(len(xticklabels)))
    for name, vals, color, marker, dashed, mfc in series:
        ax.plot(x, vals, color=color, lw=2.6,
                ls="--" if dashed else "-", marker=marker, ms=12,
                markerfacecolor=(BG if mfc == "open" else color),
                markeredgecolor=color, markeredgewidth=2.2, zorder=3)
        for xi, v in zip(x, vals):
            ax.annotate(m_log(v), (xi, v), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=13,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#dddddd", lw=0.8))
        ax.text(x[-1] + 0.06, vals[-1], name, color=color, va="center",
                ha="left", fontweight="bold", fontsize=15)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.15, len(x) - 1 + 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, color="#555555")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tokens per second (log)")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{v/1e9:.0f}B" if v >= 1e9 else f"{v/1e6:.0f}M")))
    ax.set_title(title, fontweight="bold", fontsize=16, pad=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ececec", which="both")
    # No legend: every line is labeled in-place at its right end, so a legend
    # would only overlap the data.
    fig.text(0.011, 0.02, footer, fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.11, right=0.83, top=0.9, bottom=0.14)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    # ---- Chart 1: 1 GB cl100k bar ----
    rows = [
        ("blaze-bpe (parallel)", 862.6e6, C["blaze_par"]),
        ("blaze-bpe (1 thread)", 129.5e6, C["blaze_1t"]),
        ("quicktok", 37.1e6, C["quicktok"]),
        ("rs_bpe", 12.3e6, C["rs_bpe"]),
        ("tiktoken", 5.7e6, C["tiktoken"]),
    ]
    if GT_1GB_CL:
        rows.append(("gigatoken", GT_1GB_CL, C["gigatoken"]))
    bar_chart("assets/blaze_1gb_graph.png",
              "Encoding 1 GB of Text (271.6M tokens, cl100k_base) — higher is better",
              "single 1.0 GB document, first run (cache populated during the timed pass), best of 5, same session · "
              "Apple M4 Pro, 12 cores · blaze (parallel) & gigatoken auto-parallelize; quicktok/rs_bpe/tiktoken "
              "single-thread · HF omitted (per-token Encoding needs ~16-22 GB)",
              TOK_CL, rows)

    # ---- Chart 2: 1 GB o200k bar ----
    rows = [
        ("blaze-bpe (parallel)", 573.3e6, C["blaze_par"]),
        ("blaze-bpe (1 thread)", 81.0e6, C["blaze_1t"]),
        ("quicktok", 31.4e6, C["quicktok"]),
        ("rs_bpe", 10.5e6, C["rs_bpe"]),
        ("tiktoken", 7.8e6, C["tiktoken"]),
    ]
    if GT_1GB_O2:
        rows.append(("gigatoken", GT_1GB_O2, C["gigatoken"]))
    bar_chart("assets/blaze_1gb_o200k_graph.png",
              "Encoding 1 GB of Text, o200k_base (GPT-4o/5, o1, o3) (267.8M tokens) — higher is better",
              "single 1.0 GB document, first run (cache populated during the timed pass), best of 5, same session · "
              "Apple M4 Pro, 12 cores · blaze (parallel) & gigatoken auto-parallelize; quicktok/rs_bpe/tiktoken "
              "single-thread · HF omitted (per-token Encoding needs ~16-22 GB)",
              TOK_O2, rows)

    # Sub-1 GB charts (large-documents and small-inputs lines) intentionally
    # dropped: the 1 GB regime is the representative bulk-throughput comparison,
    # and smaller sizes are dominated by fixed/per-call overheads.


if __name__ == "__main__":
    main()
