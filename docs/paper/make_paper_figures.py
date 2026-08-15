"""Generate the two cross-dataset figures used in the manuscript.

Fig. 6 — ICC(2,1) when each LLM is inserted as an additional rater, against the
         human-only baseline, for both courses.
Fig. 9 — MAE-to-human-ceiling ratio per model, OS vs Biomaterials.

Both read the CSVs already produced by the two analysis notebooks; nothing is
recomputed here. Run from the repository root:

    python docs/paper/make_paper_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "docs", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Human-only baselines reported by the two notebooks.
OS_HUMAN_ICC = 0.956
BIO_HUMAN_ICC = 0.504
OS_HUMAN_MAE = 1.113
BIO_HUMAN_MAE = 0.227

# The Biomaterials analysis keys models by raw API id; the OS analysis uses
# display labels. Map both onto one canonical label.
CANONICAL = {
    "claude-opus-4-6": "Claude Opus",
    "claude-sonnet-4-6": "Claude Sonnet",
    "claude-haiku-4-5-20251001": "Claude Haiku",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-5": "GPT-5",
    "gpt-4o": "GPT-4o",
}

# Fixed display order so the two panels stay visually comparable.
ORDER = [
    "Claude Opus",
    "Claude Sonnet",
    "Gemini 2.5 Pro",
    "Gemini 2.5 Flash",
    "Claude Haiku",
    "GPT-5",
    "GPT-4o",
]

OS_COLOR = "#2c6fbb"
BIO_COLOR = "#d97a34"


def load_frames():
    os_df = pd.read_csv(
        os.path.join(ROOT, "Grading_Dataset_OS", "outputs", "model_ranking_human_ceiling.csv")
    ).set_index("model")

    bio_icc = pd.read_csv(
        os.path.join(ROOT, "Biomaterials", "outputs", "icc_4th_grader_panel.csv")
    )
    bio_icc["model"] = bio_icc["model"].map(CANONICAL)
    bio_icc = bio_icc.set_index("model")

    bio_rank = pd.read_csv(
        os.path.join(ROOT, "Biomaterials", "outputs", "model_ranking.csv"), index_col=0
    )
    bio_rank.index = bio_rank.index.map(CANONICAL)

    return os_df, bio_icc, bio_rank


def _icc_panel(series, baseline, color, course, filename):
    """One course's insertion-test result as a standalone figure.

    Each course gets its own file: the manuscript cites this figure in that
    course's results section, and a shared two-panel figure would print the
    other course's results there too.
    """
    vals = series.reindex(ORDER)
    ypos = range(len(ORDER))

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.barh(ypos, vals.values, color=color, alpha=0.85, height=0.65)
    ax.axvline(baseline, color="black", linestyle="--", linewidth=1.5)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(ORDER, fontsize=9)
    ax.invert_yaxis()

    # Scale to this course's own baseline so the dashed line sits in a
    # comparable position across the two figures.
    ax.set_xlim(0, baseline * 1.18)
    ax.set_xlabel("ICC(2,1) with the LLM inserted as a 4th rater")
    ax.set_title(
        f"{course}: every model lowers panel reliability",
        fontsize=11.5, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    # Label the baseline on the line itself; a legend box collides with the
    # in-bar value labels wherever it is placed.
    ax.text(
        baseline + baseline * 0.022, (len(ORDER) - 1) / 2,
        f"human-only panel = {baseline:.3f}",
        rotation=90, va="center", ha="left", fontsize=8.5, style="italic",
    )

    # Labels sit inside the bars: outside, they would run across the baseline.
    for i, v in enumerate(vals.values):
        pct = 100 * v / baseline
        ax.text(v - baseline * 0.015, i, f"{v:.3f}   {pct:.0f}% of ceiling",
                va="center", ha="right", fontsize=8.5,
                color="white", fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_icc_os(os_df):
    return _icc_panel(os_df["ICC_as_4th"], OS_HUMAN_ICC, OS_COLOR,
                      "Operating Systems", "fig3_icc_as_fourth_rater_os.png")


def figure_icc_bio(bio_icc):
    return _icc_panel(bio_icc["ICC21_with_LLM"], BIO_HUMAN_ICC, BIO_COLOR,
                      "Biomaterials", "fig9_icc_as_fourth_rater_bio.png")


def figure_9(os_df, bio_rank):
    """How many times the human-human disagreement each model's error represents."""
    os_ratio = os_df["human_ratio"].reindex(ORDER)
    bio_ratio = (bio_rank["MAE_vs_humans"] / BIO_HUMAN_MAE).reindex(ORDER)

    x = range(len(ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], os_ratio.values, width,
           label="Operating Systems (ceiling = 1.113 pts/question)",
           color=OS_COLOR, alpha=0.88)
    ax.bar([i + width / 2 for i in x], bio_ratio.values, width,
           label="Biomaterials (ceiling = 0.227 pts/question)",
           color=BIO_COLOR, alpha=0.88)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
    ax.text(len(ORDER) - 0.45, 1.06, "human ceiling", fontsize=9,
            ha="right", style="italic")

    for i, (a, b) in enumerate(zip(os_ratio.values, bio_ratio.values)):
        ax.text(i - width / 2, a + 0.06, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.06, f"{b:.2f}", ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(ORDER, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Model MAE vs. humans  /  human-human MAE")
    ax.set_title(
        "No model reaches human variability in either course",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim(0, max(os_ratio.max(), bio_ratio.max()) * 1.18)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig10_cross_domain_ceiling_ratio.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    os_df, bio_icc, bio_rank = load_frames()
    print("Fig 3  ->", figure_icc_os(os_df))
    print("Fig 9  ->", figure_icc_bio(bio_icc))
    print("Fig 10 ->", figure_9(os_df, bio_rank))

    # Echo the numbers that go into the manuscript text, so the prose can be
    # checked against the figures without re-running the notebooks.
    ratio = pd.DataFrame({
        "OS_ratio": os_df["human_ratio"].reindex(ORDER),
        "BIO_ratio": (bio_rank["MAE_vs_humans"] / BIO_HUMAN_MAE).reindex(ORDER),
        "OS_ICC_4th": os_df["ICC_as_4th"].reindex(ORDER),
        "BIO_ICC_4th": bio_icc["ICC21_with_LLM"].reindex(ORDER),
    })
    print(ratio.round(3).to_string())


if __name__ == "__main__":
    main()
