"""Generate synthetic TikTok LIVE moderation datasets for analysis.

The generator writes two files:
- moderation_events.csv (event-level records)
- ccv_timeseries.csv (minute-level CCV around events)

Parameter choices are loosely calibrated to public transparency-report ratios,
then adapted to produce a manageable US-focused simulation sample.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Public-data anchors
# Source 1: 12M global LIVE closures in Q3 2024
GLOBAL_LIVE_CLOSURES_Q3_2024 = 12_000_000

# US share: historically ~8-11% of global removals (TikTok historical reports)
US_SHARE = 0.092
US_LIVE_CLOSURES_Q3_2024 = int(GLOBAL_LIVE_CLOSURES_Q3_2024 * US_SHARE)  # ~1,104,000

# Source 1: 80%+ of removals via automation
AUTOMATION_RATE = 0.82

# Source 3: 98.2% proactive removal rate in Q3 2024
PROACTIVE_REMOVAL_RATE = 0.982

# Source 2: 93.9% removed within 24 hours in Q1 2024
REMOVAL_WITHIN_24H_RATE = 0.939

# Simulation size: 4,200 events in sample, scaled back up in analysis.
N_EVENTS = 4_200
SCALE_FACTOR = US_LIVE_CLOSURES_Q3_2024 / N_EVENTS  # ~263x for impact projections

# US market parameters
CATEGORIES = ["Gaming", "Music", "Talk", "Fitness", "Education", "Commerce"]
CREATOR_TIERS = ["Nano", "Micro", "Mid", "Macro"]

# Policy categories — proportions informed by TikTok Q2 2021 public breakdown
# (Minor safety 41%, Illegal/regulated 21%, Adult nudity 14%, Violent 8%)
# Remaining split across hate speech, spam, misinformation for LIVE context
POLICY_CATEGORIES = [
    "minor_safety",
    "illegal_regulated_goods",
    "adult_nudity_sexual",
    "violent_graphic",
    "hate_speech",
    "spam_fake_engagement",
    "misinformation"
]
POLICY_WEIGHTS = [0.30, 0.18, 0.14, 0.10, 0.12, 0.10, 0.06]

INTERVENTION_TYPES = [
    "audio_mute",
    "content_warning_overlay",
    "stream_pause",
    "caption_text_removal",
    "escalated_review_flag"
]
INTERVENTION_WEIGHTS = [0.273, 0.235, 0.177, 0.151, 0.164]

SEVERITY_MAP = {
    "audio_mute":                (1.5, 2.8),
    "content_warning_overlay":   (1.2, 2.4),
    "stream_pause":              (2.8, 4.2),
    "caption_text_removal":      (1.0, 1.8),
    "escalated_review_flag":     (3.2, 4.8)
}

# Drop-off priors for each intervention type.
DROPOFF_PARAMS = {
    "audio_mute":                {"mean": 0.34, "std": 0.08},
    "content_warning_overlay":   {"mean": 0.18, "std": 0.06},
    "stream_pause":              {"mean": 0.22, "std": 0.07},
    "caption_text_removal":      {"mean": 0.12, "std": 0.05},
    "escalated_review_flag":     {"mean": 0.15, "std": 0.06}
}

# US recovery rate — lower audience loyalty vs. Asian markets
US_RECOVERY_PARAMS = {"mean": 0.11, "std": 0.04}

DETECTION_METHODS = ["automated", "human_review", "user_report"]
DETECTION_WEIGHTS = [AUTOMATION_RATE, 0.10, 0.08]

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 3, 31)


# Helpers

def random_date(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def baseline_ccv(tier):
    ranges = {
        "Nano":  (50, 400),
        "Micro": (400, 2000),
        "Mid":   (2000, 10000),
        "Macro": (10000, 80000)
    }
    lo, hi = ranges[tier]
    return int(np.random.lognormal(np.log((lo + hi) / 2), 0.4))


# Event generation

def generate_moderation_events(n=N_EVENTS):
    """
    Generate n synthetic moderation events for the US market.

    The sample is intentionally small enough to inspect quickly and is paired
    with SCALE_FACTOR so impact estimates can be projected to quarterly scale.
    """
    det_norm = [w / sum(DETECTION_WEIGHTS) for w in DETECTION_WEIGHTS]
    records = []

    for i in range(n):
        tier         = random.choices(CREATOR_TIERS, weights=[0.40, 0.30, 0.20, 0.10])[0]
        category     = random.choice(CATEGORIES)
        intervention = random.choices(INTERVENTION_TYPES, weights=INTERVENTION_WEIGHTS)[0]
        policy_cat   = random.choices(POLICY_CATEGORIES, weights=POLICY_WEIGHTS)[0]
        detection    = random.choices(DETECTION_METHODS, weights=det_norm)[0]

        sev_lo, sev_hi = SEVERITY_MAP[intervention]
        severity = round(np.random.uniform(sev_lo, sev_hi), 2)

        dropoff_params = DROPOFF_PARAMS[intervention]
        drop_off_rate = np.clip(
            np.random.normal(dropoff_params["mean"], dropoff_params["std"]),
            0.01,
            0.75,
        )
        recovery_rate = np.clip(
            np.random.normal(US_RECOVERY_PARAMS["mean"], US_RECOVERY_PARAMS["std"]),
            0.0, 0.50
        )

        baseline = baseline_ccv(tier)
        ccv_lost = int(baseline * drop_off_rate)
        ccv_recovered = int(ccv_lost * recovery_rate)

        # Keep these rates aligned with public benchmark anchors.
        is_proactive = random.random() < PROACTIVE_REMOVAL_RATE
        removed_24h = random.random() < REMOVAL_WITHIN_24H_RATE

        # A small false-positive pocket in one sensitive queue.
        is_fp = False
        if policy_cat == "hate_speech" and intervention == "escalated_review_flag":
            is_fp = random.random() < 0.08  # US estimate

        # Creator acknowledgment window for audio mutes
        creator_ack = None
        if intervention == "audio_mute":
            creator_ack = int(np.random.exponential(120)) if random.random() < 0.55 else None

        records.append({
            "event_id":             f"EVT_{i+1:05d}",
            "timestamp":            random_date(START_DATE, END_DATE).strftime("%Y-%m-%d %H:%M:%S"),
            "market":               "US",
            "creator_tier":         tier,
            "content_category":     category,
            "policy_category":      policy_cat,
            "intervention_type":    intervention,
            "detection_method":     detection,
            "severity_score":       severity,
            "is_proactive_removal": is_proactive,
            "removed_within_24h":   removed_24h,
            "is_false_positive":    is_fp,
            "baseline_ccv":         baseline,
            "ccv_lost_5min":        ccv_lost,
            "ccv_recovered_15min":  ccv_recovered,
            "drop_off_rate":        round(drop_off_rate, 4),
            "recovery_rate":        round(recovery_rate, 4),
            "creator_ack_seconds":  creator_ack,
            "scale_factor":         round(SCALE_FACTOR, 1)
        })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


# CCV time-series generation

def generate_ccv_timeseries(events_df, n_streams=300):
    """
    Build minute-by-minute CCV from T-5 to T+15 around sampled events.

    This output is mainly for charting trend shape by intervention type.
    """
    sample = events_df.sample(n=n_streams, random_state=SEED).reset_index(drop=True)
    records = []

    for _, row in sample.iterrows():
        base = row["baseline_ccv"]
        drop = row["drop_off_rate"]
        rec  = row["recovery_rate"]

        for t in range(-5, 16):
            if t < 0:
                ccv = base + int(np.random.normal(0, base * 0.03))
            elif t <= 5:
                frac = t / 5
                ccv  = base - int(base * drop * frac) + int(np.random.normal(0, base * 0.02))
            else:
                lost = base * drop
                frac = (t - 5) / 10
                ccv  = (base - lost) + int(lost * rec * frac) + int(np.random.normal(0, base * 0.02))

            records.append({
                "event_id":           row["event_id"],
                "intervention_type":  row["intervention_type"],
                "policy_category":    row["policy_category"],
                "creator_tier":       row["creator_tier"],
                "minutes_from_event": t,
                "ccv":                max(0, int(ccv))
            })

    return pd.DataFrame(records)


# Main

if __name__ == "__main__":
    print("=" * 60)
    print("TikTok LIVE Moderation Event Generator — US Market")
    print("=" * 60)
    print(f"\nReal-data anchors (TikTok Transparency Reports):")
    print(f"  Global LIVE closures Q3 2024 : {GLOBAL_LIVE_CLOSURES_Q3_2024:,}")
    print(f"  Estimated US closures Q3 2024: {US_LIVE_CLOSURES_Q3_2024:,}  (~{US_SHARE*100:.1f}% share)")
    print(f"  Automation rate              : {AUTOMATION_RATE*100:.0f}%")
    print(f"  Proactive removal rate       : {PROACTIVE_REMOVAL_RATE*100:.1f}%")
    print(f"  Removed within 24h           : {REMOVAL_WITHIN_24H_RATE*100:.1f}%")
    print(f"\n  Simulation sample : {N_EVENTS:,} events  (scale factor: ~{int(SCALE_FACTOR)}x)")

    print("\nGenerating moderation events...")
    events = generate_moderation_events()
    events.to_csv("moderation_events.csv", index=False)
    print(f"  Saved moderation_events.csv  ({len(events):,} rows)")

    print("Generating CCV time-series...")
    ts = generate_ccv_timeseries(events)
    ts.to_csv("ccv_timeseries.csv", index=False)
    print(f"  Saved ccv_timeseries.csv     ({len(ts):,} rows)")

    print("\nSample:")
    print(events[["event_id", "intervention_type", "policy_category",
                  "detection_method", "drop_off_rate", "is_proactive_removal"]].head(6).to_string(index=False))

    print("\nPolicy category distribution:")
    print(events["policy_category"].value_counts(normalize=True).mul(100).round(1).to_string())

    print("\nDetection method distribution (target: ~82% automated):")
    print(events["detection_method"].value_counts(normalize=True).mul(100).round(1).to_string())