"""Create a four-panel dashboard from the generated moderation analysis outputs.

Inputs expected in the working directory:
- moderation_events.csv
- ccv_timeseries.csv
- summary_intervention.csv
- summary_policy.csv
- summary_detection.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# Style
PALETTE = ["#4C78A8", "#72B7B2", "#A0CBE8", "#C7C7C7", "#E5E5E5"]
ACCENT_R = "#D62728"
ACCENT_G = "#2CA02C"
BG = "#FFFFFF"
GRID_C = "#E6E6E6"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.facecolor":     BG,
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.color":         GRID_C,
    "grid.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.labelsize":     10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.titlepad":      10,
})
LBL = dict(fontsize=8.5, color="#444444")
INSIGHT_BOX = dict(boxstyle="round,pad=0.35", fc="#F7F9FC", ec="#D6DDE8", alpha=0.95)

INTERVENTION_LABELS = {
    "audio_mute":               "Audio Mute",
    "stream_pause":             "Stream Pause",
    "content_warning_overlay":  "Content Warning",
    "escalated_review_flag":    "Escalated Review",
    "caption_text_removal":     "Caption Removal"
}

POLICY_LABELS = {
    "minor_safety":            "Minor Safety",
    "illegal_regulated_goods": "Illegal/Regulated",
    "adult_nudity_sexual":     "Adult Nudity",
    "violent_graphic":         "Violent/Graphic",
    "hate_speech":             "Hate Speech",
    "spam_fake_engagement":    "Spam/Fake Eng.",
    "misinformation":          "Misinformation"
}


# Panel A: drop-off by intervention type

def plot_dropoff_by_intervention(ax, df):
    df = df.copy()
    df["label"] = df["intervention_type"].map(INTERVENTION_LABELS)
    df = df.sort_values("avg_dropoff_pct", ascending=True)

    colors = [PALETTE[0] if i == len(df)-1 else PALETTE[2] for i in range(len(df))]
    bars = ax.barh(df["label"], df["avg_dropoff_pct"], color=colors, height=0.55, zorder=3)

    for bar, val in zip(bars, df["avg_dropoff_pct"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", **LBL)

    top = df.iloc[-1]
    second = df.iloc[-2]
    ratio = top["avg_dropoff_pct"] / second["avg_dropoff_pct"]
    ax.annotate(f"{ratio:.1f}× higher\nthan next type",
                xy=(top["avg_dropoff_pct"], len(df)-1),
                xytext=(top["avg_dropoff_pct"]-13, len(df)-1.65),
                fontsize=8, color=ACCENT_R,
                arrowprops=dict(arrowstyle="->", color=ACCENT_R, lw=1.2))

    ax.set_xlabel("Avg. CCV Drop-Off Rate (%) — 5 min post-event")
    ax.set_title("A  Drop-Off Rate by Intervention Type")
    ax.set_xlim(0, df["avg_dropoff_pct"].max() + 8)

    top_label = top["label"]
    insight_a = (
        "Actionable Insight:\n"
        f"Prioritize {top_label} first: it causes the steepest immediate CCV drop ({top['avg_dropoff_pct']:.1f}%)."
    )
    ax.text(0.02, 0.97, insight_a, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="#2F3E4E", bbox=INSIGHT_BOX)


# Panel B: policy-level drop-off vs recovery

def plot_policy_dropoff_vs_recovery(ax, df):
    df = df.copy()
    df["label"] = df["policy_category"].map(POLICY_LABELS)
    df = df.sort_values("avg_dropoff_pct", ascending=False)

    x = np.arange(len(df))
    w = 0.35
    b1 = ax.bar(x - w/2, df["avg_dropoff_pct"], width=w, color=ACCENT_R,
                alpha=0.85, label="Avg. Drop-Off %", zorder=3)
    b2 = ax.bar(x + w/2, df["avg_recovery_pct"], width=w, color=ACCENT_G,
                alpha=0.85, label="Avg. Recovery %", zorder=3)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", fontsize=7.5, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Rate (%)")
    ax.set_title("B  Drop-Off vs. Recovery by Policy Category")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_ylim(0, df[["avg_dropoff_pct","avg_recovery_pct"]].max().max() + 7)

    largest_volume = df.loc[df["event_count"].idxmax()]
    largest_per_event = df.loc[df["avg_dropoff_pct"].idxmax()]
    insight_b = (
        "Actionable Insight:\n"
        f"Ops priority: {largest_volume['label']} (largest volume); QA priority: {largest_per_event['label']} (highest per-event drop-off)."
    )
    ax.text(0.02, 0.97, insight_b, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="#2F3E4E", bbox=INSIGHT_BOX)


# Panel C: CCV trajectory by intervention

def plot_ccv_trajectory(ax, ts_df):
    # Normalize each stream to its own pre-event baseline so trends are comparable.
    pre = ts_df[ts_df["minutes_from_event"] == -1][["event_id","ccv"]].rename(columns={"ccv":"base"})
    ts = ts_df.merge(pre, on="event_id", how="left")
    ts["ccv_pct"] = ts["ccv"] / ts["base"].replace(0, np.nan) * 100

    colors = PALETTE[:5]
    for (itype, color) in zip(
        ["audio_mute","stream_pause","content_warning_overlay",
         "escalated_review_flag","caption_text_removal"], colors):
        sub = ts[ts["intervention_type"] == itype]
        g   = sub.groupby("minutes_from_event")["ccv_pct"].agg(["mean","std"]).reset_index()
        label = INTERVENTION_LABELS[itype]
        ax.plot(g["minutes_from_event"], g["mean"], color=color, lw=2, label=label, zorder=3)
        ax.fill_between(g["minutes_from_event"],
                        g["mean"]-g["std"], g["mean"]+g["std"],
                        color=color, alpha=0.08, zorder=2)

    ax.axvline(0, color="#888", lw=1.2, ls="--", zorder=4)
    ax.text(0.3, 72, "Moderation\nevent", fontsize=8, color="#666")
    ax.set_xlabel("Minutes from Moderation Event")
    ax.set_ylabel("CCV (% of pre-event baseline)")
    ax.set_title("C  CCV Trajectory by Intervention Type")
    ax.legend(fontsize=7.5, framealpha=0.7, ncol=2)
    ax.set_xlim(-5, 15)
    ax.set_ylim(55, 108)

    insight_c = (
        "Actionable Insight:\n"
        "Tighten the 5-10 minute post-intervention recovery loop to minimize retention decay in LIVE sessions."
    )
    ax.text(0.02, 0.05, insight_c, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8, color="#2F3E4E", bbox=INSIGHT_BOX)
    ax.text(0.98, 0.05,
            "Lightweight rigor: shaded bands show +/-1 std dev\n(across streams, descriptive only)",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5, color="#5A6775")


# Panel D: detection mix vs benchmark

def plot_detection_mix(ax, df):
    labels = df["detection_method"].tolist()
    x = np.arange(len(labels))
    bars = ax.bar(x, df["pct_of_total"], color=PALETTE[:len(labels)],
                  width=0.5, zorder=3)

    for bar, val in zip(bars, df["pct_of_total"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", **LBL)

    # Reference line from the published automation benchmark.
    ax.axhline(82, color=ACCENT_R, lw=1.5, ls="--", zorder=5)
    ax.text(len(labels)-0.45, 83.2, "Real benchmark: 82% automated\n(TikTok Transparency Report, Dec 2024)",
            fontsize=7.5, color=ACCENT_R, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(["Automated", "Human Review", "User Report"], fontsize=10)
    ax.set_ylabel("% of Total Moderation Events")
    ax.set_title("D  Detection Method Distribution vs. Real Benchmark")
    ax.set_ylim(0, 100)

    automated = df.loc[df["detection_method"] == "automated"]
    human = df.loc[df["detection_method"] == "human_review"]
    if not automated.empty and not human.empty:
        drop_gap = automated.iloc[0]["avg_dropoff_pct"] - human.iloc[0]["avg_dropoff_pct"]
        insight_d = (
            "Actionable Insight:\n"
            f"Keep automation for scale, then fast-track borderline cases to human review (drop-off gap: {drop_gap:.1f}pp)."
        )
    else:
        insight_d = (
            "Actionable Insight:\n"
            "Keep proactive automation high and target human QA where creator-facing false-positive risk is highest."
        )
    ax.text(0.02, 0.97, insight_d, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="#2F3E4E", bbox=INSIGHT_BOX)


# Dashboard composition

def build_dashboard():
    events = pd.read_csv("moderation_events.csv")
    ts_df = pd.read_csv("ccv_timeseries.csv")
    summary_intervention = pd.read_csv("summary_intervention.csv")
    summary_policy = pd.read_csv("summary_policy.csv")
    summary_detection = pd.read_csv("summary_detection.csv")
    summary_impact = pd.read_csv("summary_impact.csv")

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.suptitle(
        "TikTok LIVE — Engagement Drop-Off in Moderated Streams (US Market)\n"
        "Q1 2026 Synthetic Analysis  |  Grounded in TikTok Transparency Reports  |  Akbota Kengeskhan",
        fontsize=13, fontweight="bold", color="#1F3864", y=0.97
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.50, wspace=0.32,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_dropoff_by_intervention(ax1, summary_intervention)
    plot_policy_dropoff_vs_recovery(ax2, summary_policy)
    plot_ccv_trajectory(ax3, ts_df)
    plot_detection_mix(ax4, summary_detection)

    top_priority = summary_impact.sort_values("Projected Q1 Recoverable CCV", ascending=False).iloc[0]
    fig.text(
        0.07,
        0.055,
        "Priority Recommendation: "
        f"{top_priority['Solution']} (Projected recoverable CCV: {int(top_priority['Projected Q1 Recoverable CCV']):,})",
        fontsize=8.2,
        color="#2F3E4E",
        fontweight="bold",
    )

    # Footnote with source context and modeling caveat.
    fig.text(0.07, 0.01,
             "Data sources: TikTok Community Guidelines Enforcement Reports (Q1 2024, Q3 2024, Q1 2025); "
             "TikTok submission to UK Parliament (Jan 2025).\n"
             "CCV drop-off and recovery rates are synthetic extrapolations calibrated to published structural ratios. "
             "Multipliers in impact sizing are conservative estimates based on observed deltas. "
             "These charts are descriptive and should not be interpreted as causal inference.",
             fontsize=7, color="#888888", style="italic")

    plt.savefig("engagement_dropoff_dashboard.png", dpi=160, bbox_inches="tight")
    print("Saved: engagement_dropoff_dashboard.png")
    plt.close()


if __name__ == "__main__":
    print("Building dashboard...")
    build_dashboard()
    print("Done.")