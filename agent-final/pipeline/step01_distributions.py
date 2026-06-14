"""
Step 01 — Distribution & data quality audit (MUST pass before any modeling).

Checks:
  - Per-column dtype, missing, unique, constant columns
  - Numeric: mean, std, skew, kurtosis, zero%, IQR outliers
  - Categorical: value counts, cardinality
  - Target Usage_kWh: distribution shape, zeros, spikes
  - Temporal: frequency, gaps, coverage by month
  - Duplicate timestamps

Run: python pipeline/step01_distributions.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step01"
RAW = ROOT.parent / "Steel_industry_data.csv"
TARGET = "Usage_kWh"

NUMERIC = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]
CATEGORICAL = ["WeekStatus", "Day_of_week", "Load_Type"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    return df.sort_values("date").reset_index(drop=True)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC:
        s = df[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        rows.append({
            "column": col,
            "count": int(s.count()),
            "missing": int(s.isna().sum()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "p25": round(float(q1), 4),
            "median": round(float(s.median()), 4),
            "p75": round(float(q3), 4),
            "max": round(float(s.max()), 4),
            "skew": round(float(stats.skew(s.dropna())), 4),
            "kurtosis": round(float(stats.kurtosis(s.dropna())), 4),
            "zero_pct": round(100 * (s == 0).mean(), 2),
            "outlier_iqr_n": int(outlier),
            "outlier_iqr_pct": round(100 * outlier / len(s), 2),
            "is_constant": bool(s.nunique() <= 1),
            "n_unique": int(s.nunique()),
        })
    return pd.DataFrame(rows)


def categorical_profile(df: pd.DataFrame) -> dict:
    prof = {}
    for col in CATEGORICAL:
        vc = df[col].value_counts()
        prof[col] = {
            "n_unique": int(df[col].nunique()),
            "counts": {str(k): int(v) for k, v in vc.items()},
            "missing": int(df[col].isna().sum()),
        }
    return prof


def temporal_audit(df: pd.DataFrame) -> dict:
    diffs = df["date"].diff().dropna()
    mode = diffs.mode().iloc[0]
    gaps = diffs[diffs != mode]
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    per_month = df.groupby("month").size().to_dict()
    return {
        "start": str(df["date"].min()),
        "end": str(df["date"].max()),
        "n_rows": len(df),
        "expected_15min_rows_per_year": 365 * 24 * 4,
        "mode_interval_minutes": int(mode.total_seconds() / 60),
        "irregular_gaps": int(len(gaps)),
        "duplicate_timestamps": int(df["date"].duplicated().sum()),
        "rows_per_month": per_month,
    }


def decide_next_step(num_prof: pd.DataFrame, cat_prof: dict, temporal: dict, df: pd.DataFrame) -> dict:
    """Human-readable decisions for step 02 based on step 01 results."""
    d = {}

    # Distribution issues
    skewed = num_prof[num_prof["skew"].abs() > 2]["column"].tolist()
    d["highly_skewed_features"] = skewed
    d["skew_action"] = "log1p candidate for right-skewed inputs (not target yet — decide in step 03)"

    zero_heavy = num_prof[num_prof["zero_pct"] > 30]["column"].tolist()
    d["zero_inflated_features"] = zero_heavy
    d["zero_action"] = "flag CO2/Leading reactive — may carry little signal when zero; check leakage in step 03"

    # NSM is deterministic time counter
    nsm_unique = num_prof.loc[num_prof["column"] == "NSM", "n_unique"].iloc[0]
    d["nsm_note"] = f"NSM has {nsm_unique} unique values — likely redundant with datetime (900-step counter)"
    d["nsm_action"] = "compare correlation with hour-of-day in step 02; likely DROP or replace with hour"

    # Day_of_week vs WeekStatus
    d["day_week_note"] = "Day_of_week (7 levels) overlaps WeekStatus (2 levels) — redundancy check in step 02"

    # Target
    t = df[TARGET]
    d["target_skew"] = round(float(stats.skew(t)), 4)
    d["target_zero_pct"] = round(100 * (t == 0).mean(), 2)
    d["target_action"] = "right-skewed target — keep raw for interpretability; scaler handles range"

    # Quality gates
    d["quality_pass"] = (
        temporal["irregular_gaps"] == 0
        and temporal["duplicate_timestamps"] == 0
        and num_prof["missing"].sum() == 0
    )
    d["next_step"] = "step02_correlation — feature-feature r, VIF, target correlation, leakage flags"
    return d


def plot_distributions(df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1 — All numeric histograms (2x4 grid)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, NUMERIC):
        s = df[col]
        ax.hist(s, bins=50, edgecolor="white", alpha=0.85)
        ax.set_title(f"{col}\nskew={stats.skew(s):.2f}, zeros={100*(s==0).mean():.0f}%", fontsize=8)
    fig.suptitle("Step 01 — Numeric feature distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01_numeric_histograms.png", dpi=130)
    plt.close(fig)

    # 2 — Log-scale for skewed non-target
    skew_cols = [c for c in NUMERIC if c != TARGET and stats.skew(df[c]) > 1]
    if skew_cols:
        n = len(skew_cols)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, skew_cols):
            v = df[col][df[col] > 0]
            ax.hist(np.log1p(v), bins=40, edgecolor="white")
            ax.set_title(f"log1p({col})")
        fig.tight_layout()
        fig.savefig(OUT / "02_log_transform_preview.png", dpi=130)
        plt.close(fig)

    # 3 — Target deep dive: hist + KDE + QQ
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    t = df[TARGET]
    axes[0].hist(t, bins=60, edgecolor="white", density=True)
    t.plot(kind="kde", ax=axes[0], color="red")
    axes[0].set_title(f"{TARGET} — histogram + KDE")
    stats.probplot(t, dist="norm", plot=axes[1])
    axes[1].set_title(f"{TARGET} — Q-Q vs normal")
    sns.boxplot(y=t, ax=axes[2])
    axes[2].set_title(f"{TARGET} — boxplot")
    fig.tight_layout()
    fig.savefig(OUT / "03_target_distribution_deep.png", dpi=130)
    plt.close(fig)

    # 4 — Categorical bar charts
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, CATEGORICAL):
        vc = df[col].value_counts()
        vc.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "04_categorical_counts.png", dpi=130)
    plt.close(fig)

    # 5 — Target by category (distribution comparison)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, CATEGORICAL):
        sns.boxplot(data=df, x=col, y=TARGET, ax=ax)
        ax.set_title(f"{TARGET} by {col}")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "05_target_by_category.png", dpi=130)
    plt.close(fig)

    # 6 — Temporal: rows per month + target monthly mean
    df2 = df.copy()
    df2["month"] = df2["date"].dt.to_period("M").astype(str)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    df2.groupby("month").size().plot(kind="bar", ax=axes[0], color="gray")
    axes[0].set_title("Row count per month (should be ~2880 for full months)")
    df2.groupby("month")[TARGET].mean().plot(kind="line", marker="o", ax=axes[1])
    axes[1].set_title(f"Mean {TARGET} per month")
    fig.tight_layout()
    fig.savefig(OUT / "06_temporal_coverage.png", dpi=130)
    plt.close(fig)

    # 7 — Hour-of-day target pattern
    df2["hour"] = df2["date"].dt.hour
    fig, ax = plt.subplots(figsize=(10, 4))
    df2.groupby("hour")[TARGET].agg(["mean", "std"]).plot(ax=ax)
    ax.set_title("Usage_kWh mean ± std by hour of day")
    fig.tight_layout()
    fig.savefig(OUT / "07_hourly_pattern.png", dpi=130)
    plt.close(fig)


def main():
    print("=" * 60)
    print("STEP 01 — Distribution & data quality audit")
    print("=" * 60)

    df = load_raw()
    num_prof = numeric_profile(df)
    cat_prof = categorical_profile(df)
    temporal = temporal_audit(df)
    decisions = decide_next_step(num_prof, cat_prof, temporal, df)

    plot_distributions(df)

    report = {
        "numeric_profile": num_prof.to_dict(orient="records"),
        "categorical_profile": cat_prof,
        "temporal": temporal,
        "decisions": decisions,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step01_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    num_prof.to_csv(OUT / "numeric_profile.csv", index=False)

    # DECISIONS.md for human review before step 02
    lines = [
        "# Step 01 — Results & decisions for Step 02",
        "",
        f"**Quality gate:** {'PASS' if decisions['quality_pass'] else 'FAIL'}",
        "",
        "## Temporal",
        f"- {temporal['n_rows']} rows, {temporal['start']} → {temporal['end']}",
        f"- Interval: {temporal['mode_interval_minutes']} min, gaps: {temporal['irregular_gaps']}, dupes: {temporal['duplicate_timestamps']}",
        "",
        "## Numeric distribution summary",
        num_prof.to_markdown(index=False),
        "",
        "## Distribution flags",
        f"- **Highly skewed (|skew|>2):** {decisions['highly_skewed_features'] or 'none'}",
        f"- **Zero-inflated (>30% zeros):** {decisions['zero_inflated_features']}",
        f"- **Target skew:** {decisions['target_skew']}, target zeros: {decisions['target_zero_pct']}%",
        "",
        "## Redundancy hints (investigate in Step 02)",
        f"- {decisions['nsm_note']}",
        f"- {decisions['day_week_note']}",
        "",
        "## Actions decided",
        f"- Skew: {decisions['skew_action']}",
        f"- Zeros: {decisions['zero_action']}",
        f"- NSM: {decisions['nsm_action']}",
        f"- Target: {decisions['target_action']}",
        "",
        "## Next step",
        f"**{decisions['next_step']}**",
        "",
        "Plots: `outputs/step01/01_*.png` … `07_*.png`",
    ]
    (OUT / "DECISIONS.md").write_text("\n".join(lines), encoding="utf-8")

    print(num_prof.to_string(index=False))
    print(f"\nQuality pass: {decisions['quality_pass']}")
    print(f"Zero-inflated: {decisions['zero_inflated_features']}")
    print(f"Highly skewed: {decisions['highly_skewed_features']}")
    print(f"\nWrote {OUT}/DECISIONS.md — READ BEFORE step 02")


if __name__ == "__main__":
    main()
