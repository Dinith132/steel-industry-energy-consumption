"""
Step 02 — Correlation & redundancy screening.

Reads step01 report. Writes drop list for step03.

Run: python pipeline/step02_correlation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step02"
STEP01 = ROOT / "outputs" / "step01"
RAW = ROOT.parent / "energydata_complete.csv"
TARGET = "Appliances"

HIGH_R_THRESHOLD = 0.85
LEAKAGE_R_THRESHOLD = 0.95
VIF_DROP_THRESHOLD = 10.0

# Must drop — random UCI columns
MUST_DROP = ["rv1", "rv2"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["date"])
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from numpy.linalg import LinAlgError

    X = df[cols].astype(float).values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    vifs = []
    for i, col in enumerate(cols):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        others = np.column_stack([np.ones(len(y)), others])
        try:
            beta, _, _, _ = np.linalg.lstsq(others, y, rcond=None)
            pred = others @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else np.inf
        except LinAlgError:
            vif = np.inf
        vifs.append({"feature": col, "VIF": round(float(vif), 2)})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False)


def high_corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "r": round(float(r), 4)})
    return pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)


def decide_features(corr_target: pd.Series, ff_pairs: pd.DataFrame, vif_df: pd.DataFrame) -> dict:
    drop = set(MUST_DROP)
    drop_reason = {
        "rv1": "Random UCI placeholder — identical to rv2, no physical meaning",
        "rv2": "Random UCI placeholder — drop with rv1",
    }

    # Redundancy rules from high-corr pairs
    for _, row in ff_pairs.iterrows():
        a, b, r = row["feature_a"], row["feature_b"], row["r"]
        if {a, b} == {"T6", "T_out"}:
            drop.add("T6")
            drop_reason["T6"] = f"Near-duplicate of T_out (r={r}) — keep outdoor T_out"
        if "T9" in (a, b) and abs(r) > 0.9:
            drop.add("T9")
            drop_reason["T9"] = f"Highly correlated with other room sensors (r up to {r})"

    input_cols = [c for c in corr_target.index if c not in drop and c != TARGET]
    keep = [TARGET] + [c for c in input_cols if c not in drop]

    leakage = {k: round(float(v), 4) for k, v in corr_target.items() if abs(v) >= LEAKAGE_R_THRESHOLD}

    return {
        "leakage_candidates": leakage,
        "proposed_drop": sorted(drop),
        "drop_reason": drop_reason,
        "proposed_keep": sorted(set(keep) - {TARGET}),
        "proposed_engineered": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
        "high_vif": vif_df[vif_df["VIF"] >= VIF_DROP_THRESHOLD].to_dict(orient="records"),
        "next_step": "step03_preprocess — apply drops, engineer time, save 10min + hourly tracks",
    }


def main():
    print("=" * 60)
    print("STEP 02 — Correlation screening")
    print("=" * 60)

    step01 = json.loads((STEP01 / "step01_report.json").read_text())
    if not step01.get("decisions", {}).get("quality_pass"):
        print("WARNING: step01 quality gate did not pass")

    df = load_raw()
    OUT.mkdir(parents=True, exist_ok=True)

    input_cols = [c for c in df.columns if c not in ("date", TARGET)]
    corr_ff = df[input_cols].corr()
    pairs = high_corr_pairs(corr_ff, HIGH_R_THRESHOLD)

    all_num = [TARGET] + input_cols
    corr_target = df[all_num].corr()[TARGET].drop(TARGET)
    corr_target = corr_target.loc[corr_target.abs().sort_values(ascending=False).index]

    vif_cols = [c for c in input_cols if c not in MUST_DROP]
    vif_df = compute_vif(df, vif_cols)

    decisions = decide_features(corr_target, pairs, vif_df)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_ff, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature-feature correlation (raw inputs)")
    fig.tight_layout()
    fig.savefig(OUT / "01_feature_feature_corr.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_target.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Correlation with {TARGET}")
    fig.tight_layout()
    fig.savefig(OUT / "02_target_correlation.png", dpi=120)
    plt.close(fig)

    report = {
        "high_corr_pairs": pairs.to_dict(orient="records"),
        "target_correlation": {k: round(float(v), 4) for k, v in corr_target.items()},
        "vif": vif_df.to_dict(orient="records"),
        "decisions": decisions,
    }
    (OUT / "step02_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pairs.to_csv(OUT / "high_corr_pairs.csv", index=False)
    corr_target.to_csv(OUT / "target_correlation.csv")

    lines = [
        "# Step 02 — Correlation decisions",
        "",
        "## Leakage (|r| >= 0.95 with target)",
        str(decisions["leakage_candidates"] or "none — good"),
        "",
        "## High corr pairs (|r| >= 0.85)",
        pairs.to_markdown(index=False) if len(pairs) else "_none_",
        "",
        "## DROP list",
        "```",
        str(decisions["proposed_drop"]),
        "```",
        "",
        "| Feature | Reason |",
        "|---------|--------|",
    ]
    for f in decisions["proposed_drop"]:
        lines.append(f"| {f} | {decisions['drop_reason'].get(f, 'redundant')} |")
    lines += [
        "",
        "## KEEP inputs",
        str(decisions["proposed_keep"]),
        "",
        "## Engineered",
        str(decisions["proposed_engineered"]),
        "",
        "## Next",
        f"**{decisions['next_step']}**",
    ]
    (OUT / "DECISIONS.md").write_text("\n".join(lines), encoding="utf-8")

    print("Top target correlations:")
    print(corr_target.head())
    print(f"\nDROP: {decisions['proposed_drop']}")
    print(f"Wrote {OUT}/DECISIONS.md")


if __name__ == "__main__":
    main()
