"""
Step 01 — Distribution & data quality audit for energydata_complete.csv.

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
RAW = ROOT.parent / "energydata_complete.csv"
TARGET = "Appliances"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["date"])
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == "date":
            continue
        s = df[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        rows.append({
            "column": col,
            "missing": int(s.isna().sum()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "median": round(float(s.median()), 4),
            "max": round(float(s.max()), 4),
            "skew": round(float(stats.skew(s.dropna())), 4),
            "zero_pct": round(100 * (s == 0).mean(), 2),
            "n_unique": int(s.nunique()),
        })
    return pd.DataFrame(rows)


def temporal_audit(df: pd.DataFrame) -> dict:
    diffs = df["date"].diff().dropna()
    mode = diffs.mode().iloc[0]
    gaps = diffs[diffs != mode]
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return {
        "start": str(df["date"].min()),
        "end": str(df["date"].max()),
        "n_rows": len(df),
        "mode_interval_minutes": int(mode.total_seconds() / 60),
        "irregular_gaps": int(len(gaps)),
        "duplicate_timestamps": int(df["date"].duplicated().sum()),
        "rows_per_month": df.groupby("month").size().to_dict(),
    }


def decide_next_step(num_prof: pd.DataFrame, temporal: dict, df: pd.DataFrame) -> dict:
    skewed = num_prof[num_prof["skew"].abs() > 2]["column"].tolist()
    rv_same = bool((df["rv1"] == df["rv2"]).all()) if "rv1" in df.columns else False
    return {
        "quality_pass": (
            temporal["irregular_gaps"] == 0
            and temporal["duplicate_timestamps"] == 0
            and num_prof["missing"].sum() == 0
        ),
        "highly_skewed_features": skewed,
        "target_skew": round(float(stats.skew(df[TARGET])), 4),
        "target_zero_pct": round(100 * (df[TARGET] == 0).mean(), 2),
        "rv1_equals_rv2": rv_same,
        "rv_note": "rv1 and rv2 are identical random UCI placeholders — drop in step02",
        "null_action": "No nulls after strip — no imputation needed",
        "next_step": "step02_correlation — redundancy, target corr, drop list",
    }


def plot_distributions(df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    num_cols = [c for c in df.columns if c != "date"]

    # histogram grid (skip date)
    n = len(num_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
    axes = np.array(axes).ravel()
    for ax, col in zip(axes, num_cols):
        s = df[col]
        ax.hist(s, bins=40, edgecolor="white", alpha=0.85)
        ax.set_title(f"{col}\nskew={stats.skew(s):.2f}", fontsize=7)
    for ax in axes[len(num_cols):]:
        ax.axis("off")
    fig.suptitle("Step 01 — Column distributions")
    fig.tight_layout()
    fig.savefig(OUT / "01_numeric_histograms.png", dpi=120)
    plt.close(fig)

    # target deep dive
    t = df[TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(t, bins=60, edgecolor="white")
    axes[0].set_title(f"{TARGET} histogram (skew={stats.skew(t):.2f})")
    df2 = df.copy()
    df2["hour"] = df2["date"].dt.hour
    df2.groupby("hour")[TARGET].mean().plot(ax=axes[1], marker="o")
    axes[1].set_title("Mean Appliances by hour")
    fig.tight_layout()
    fig.savefig(OUT / "02_target_and_hourly_pattern.png", dpi=120)
    plt.close(fig)

    # temporal coverage
    df2["month"] = df2["date"].dt.to_period("M").astype(str)
    fig, ax = plt.subplots(figsize=(10, 4))
    df2.groupby("month").size().plot(kind="bar", ax=ax, color="gray")
    ax.set_title("Rows per month (10-min grid)")
    fig.tight_layout()
    fig.savefig(OUT / "03_temporal_coverage.png", dpi=120)
    plt.close(fig)


def main():
    print("=" * 60)
    print("STEP 01 — Distribution audit (Appliances dataset)")
    print("=" * 60)

    df = load_raw()
    num_prof = numeric_profile(df)
    temporal = temporal_audit(df)
    decisions = decide_next_step(num_prof, temporal, df)
    plot_distributions(df)

    report = {"numeric_profile": num_prof.to_dict(orient="records"), "temporal": temporal, "decisions": decisions}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step01_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    num_prof.to_csv(OUT / "numeric_profile.csv", index=False)

    lines = [
        "# Step 01 — Results",
        "",
        f"**Quality gate:** {'PASS' if decisions['quality_pass'] else 'FAIL'}",
        "",
        f"- Rows: {temporal['n_rows']}, {temporal['start']} → {temporal['end']}",
        f"- Interval: {temporal['mode_interval_minutes']} min, gaps: {temporal['irregular_gaps']}",
        f"- Null handling: {decisions['null_action']}",
        f"- Target skew: {decisions['target_skew']}, zeros: {decisions['target_zero_pct']}%",
        f"- rv1==rv2 on all rows: {decisions['rv1_equals_rv2']} → {decisions['rv_note']}",
        "",
        "## Highly skewed (|skew|>2)",
        str(decisions["highly_skewed_features"]),
        "",
        "## Next",
        f"**{decisions['next_step']}**",
    ]
    (OUT / "DECISIONS.md").write_text("\n".join(lines), encoding="utf-8")

    print(num_prof.head(10).to_string(index=False))
    print(f"\nQuality pass: {decisions['quality_pass']}")
    print(f"Wrote {OUT}/DECISIONS.md")


if __name__ == "__main__":
    main()
