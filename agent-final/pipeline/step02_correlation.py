"""
Step 02 — Correlation, multicollinearity (VIF), and target-leakage screening.

Reads step01 decisions first. Produces:
  - Feature-feature correlation heatmap + high-r pairs table
  - Correlation with target (leakage candidates if |r| > 0.95)
  - VIF on candidate INPUT features
  - NSM vs derived hour redundancy check
  - Day_of_week vs WeekStatus redundancy

Run AFTER step01: python pipeline/step02_correlation.py
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
INPUT_NUMERIC = [c for c in NUMERIC if c != TARGET]
CATEGORICAL = ["WeekStatus", "Day_of_week", "Load_Type"]

HIGH_R_THRESHOLD = 0.85
LEAKAGE_R_THRESHOLD = 0.95
VIF_DROP_THRESHOLD = 10.0


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("date").reset_index(drop=True)
    df["hour"] = df["date"].dt.hour
    df["minute_of_day"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    return df


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """VIF via OLS — requires statsmodels or manual."""
    from numpy.linalg import LinAlgError

    X = df[cols].astype(float).values
    # standardize for numerical stability
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    n = X.shape[1]
    vifs = []
    for i in range(n):
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
        vifs.append({"feature": cols[i], "VIF": round(float(vif), 2)})
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


def encode_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL:
        out[col + "_code"] = pd.Categorical(out[col]).codes
    return out


def decide_features(
    corr_target: pd.Series,
    ff_pairs: pd.DataFrame,
    vif_df: pd.DataFrame,
    nsm_hour_r: float,
    dow_week_r: float,
    step01: dict,
) -> dict:
    """Feature keep/drop/engineer decisions for step 03."""
    d = {}

    # Leakage: features almost identical to target (not usable as honest predictors)
    leakage = corr_target[corr_target.index != TARGET]
    leakage = leakage[leakage.abs() >= LEAKAGE_R_THRESHOLD]
    d["leakage_candidates"] = {k: round(float(v), 4) for k, v in leakage.items()}
    d["leakage_action"] = {}
    for feat in leakage.index:
        if feat == "CO2(tCO2)":
            d["leakage_action"][feat] = "DROP — 0.99 corr with Usage_kWh; CO2 is computed FROM energy use (target leakage)"
        elif feat == "Lagging_Current_Reactive.Power_kVarh":
            d["leakage_action"][feat] = "REVIEW — 0.90 corr; physically coupled but not deterministic leakage; KEEP with caution or use lag-only in window"
        else:
            d["leakage_action"][feat] = "DROP or justify — near-perfect target correlation"

    # Redundant pairs
    drop_from_redundancy = set()
    redundant_notes = []
    for _, row in ff_pairs.iterrows():
        a, b, r = row["feature_a"], row["feature_b"], row["r"]
        note = f"{a} ↔ {b}: r={r}"
        redundant_notes.append(note)
        # Specific domain rules
        if {a, b} == {"Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"} and abs(r) > 0.5:
            redundant_notes.append("  → power factors partially redundant; consider keeping lagging only")
        if "NSM" in (a, b) and "hour" in (a, b):
            redundant_notes.append("  → NSM duplicates time; DROP NSM, use hour")

    d["high_corr_pairs"] = redundant_notes
    d["nsm_vs_hour_r"] = round(nsm_hour_r, 4)
    d["nsm_decision"] = "DROP NSM" if abs(nsm_hour_r) > 0.9 else "keep NSM"
    d["dow_vs_weekstatus_r"] = round(dow_week_r, 4)
    d["dow_decision"] = "DROP Day_of_week (keep WeekStatus)" if abs(dow_week_r) > 0.7 else "keep both"

    # VIF
    high_vif = vif_df[vif_df["VIF"] >= VIF_DROP_THRESHOLD]
    d["high_vif_features"] = high_vif.to_dict(orient="records")
    d["vif_action"] = "Drop or combine features with VIF >= 10 before modeling"

    # Zero-inflated from step01
    zero_feats = step01.get("decisions", {}).get("zero_inflated_features", [])
    d["zero_inflated"] = zero_feats
    d["zero_action"] = {
        "Leading_Current_Reactive_Power_kVarh": "DROP — 67% zeros, weak negative corr (-0.32), little information",
        "CO2(tCO2)": "DROP — leakage + 60% zeros",
    }

    # Proposed feature set for modeling
    drop = set()
    drop.add("CO2(tCO2)")
    drop.add("NSM")
    drop.add("Leading_Current_Reactive_Power_kVarh")
    if d["dow_decision"].startswith("DROP"):
        drop.add("Day_of_week")

    keep_numeric = [c for c in INPUT_NUMERIC if c not in drop]
    keep_cat = [c for c in CATEGORICAL if c not in drop]
    d["proposed_drop"] = sorted(drop)
    d["proposed_keep_numeric"] = keep_numeric
    d["proposed_keep_categorical"] = keep_cat
    d["proposed_engineered"] = ["hour", "day_of_week_sin", "day_of_week_cos"]  # if drop Day_of_week, use sin/cos hour only
    d["next_step"] = "step03_feature_selection — confirm drops, apply preprocessing, save clean artifacts"
    return d


def main():
    print("=" * 60)
    print("STEP 02 — Correlation, VIF, leakage screening")
    print("=" * 60)

    step01 = json.loads((STEP01 / "step01_report.json").read_text())
    if not step01.get("decisions", {}).get("quality_pass"):
        print("WARNING: Step 01 quality gate did not pass")

    df = load_raw()
    enc = encode_for_analysis(df)
    OUT.mkdir(parents=True, exist_ok=True)

    # Feature-feature (inputs only, no target)
    input_cols = INPUT_NUMERIC + [c + "_code" for c in CATEGORICAL] + ["hour"]
    corr_ff = enc[input_cols].corr()
    pairs = high_corr_pairs(corr_ff, HIGH_R_THRESHOLD)

    # Target correlation (all numerics + encoded cats)
    all_for_target = NUMERIC + [c + "_code" for c in CATEGORICAL]
    ct = enc[all_for_target + [TARGET]].corr()[TARGET]
    corr_target = ct.drop(TARGET, errors="ignore")
    if isinstance(corr_target, pd.DataFrame):
        corr_target = corr_target.iloc[:, 0]
    corr_target = corr_target.loc[corr_target.abs().sort_values(ascending=False).index]

    # VIF on input numerics (excluding obvious drops for VIF of remaining)
    vif_cols = [c for c in INPUT_NUMERIC if c not in ("CO2(tCO2)", "NSM", "Leading_Current_Reactive_Power_kVarh")]
    vif_df = compute_vif(enc, vif_cols)

    # NSM vs hour
    nsm_hour_r = enc["NSM"].corr(enc["hour"])

    # Day_of_week vs WeekStatus (encoded)
    dow_week_r = enc["Day_of_week_code"].corr(enc["WeekStatus_code"])

    decisions = decide_features(corr_target, pairs, vif_df, nsm_hour_r, dow_week_r, step01)

    # Plots
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_ff, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature-feature correlation (inputs + hour)")
    fig.tight_layout()
    fig.savefig(OUT / "01_feature_feature_corr.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    corr_target.plot(kind="barh", ax=ax, color=["red" if abs(v) >= LEAKAGE_R_THRESHOLD else "steelblue" for v in corr_target])
    ax.axvline(LEAKAGE_R_THRESHOLD, color="red", ls="--", label="leakage threshold 0.95")
    ax.axvline(-LEAKAGE_R_THRESHOLD, color="red", ls="--")
    ax.set_title("Correlation with Usage_kWh (target)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "02_target_correlation.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    vif_df.plot(x="feature", y="VIF", kind="bar", ax=ax, legend=False, color="teal")
    ax.axhline(VIF_DROP_THRESHOLD, color="red", ls="--", label=f"VIF={VIF_DROP_THRESHOLD}")
    ax.set_title("Variance Inflation Factor (input features)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "03_vif.png", dpi=130)
    plt.close(fig)

    # Scatter: CO2 vs Usage (show leakage visually)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sample = enc.sample(min(3000, len(enc)), random_state=42)
    axes[0].scatter(sample["CO2(tCO2)"], sample[TARGET], s=5, alpha=0.3)
    axes[0].set_xlabel("CO2"); axes[0].set_ylabel("Usage_kWh")
    axes[0].set_title(f"CO2 vs target (r={corr_target['CO2(tCO2)']:.3f}) — LEAKAGE")
    axes[1].scatter(sample["NSM"], sample["hour"], s=5, alpha=0.3)
    axes[1].set_xlabel("NSM"); axes[1].set_ylabel("hour")
    axes[1].set_title(f"NSM vs hour (r={nsm_hour_r:.3f}) — redundant")
    fig.tight_layout()
    fig.savefig(OUT / "04_leakage_redundancy_scatter.png", dpi=130)
    plt.close(fig)

    report = {
        "high_corr_pairs": pairs.to_dict(orient="records"),
        "target_correlation": {k: round(float(v), 4) for k, v in corr_target.items()},
        "vif": vif_df.to_dict(orient="records"),
        "nsm_hour_r": round(float(nsm_hour_r), 4),
        "dow_weekstatus_r": round(float(dow_week_r), 4),
        "decisions": decisions,
    }
    (OUT / "step02_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pairs.to_csv(OUT / "high_corr_pairs.csv", index=False)
    corr_target.to_csv(OUT / "target_correlation.csv")

    lines = [
        "# Step 02 — Correlation & leakage decisions",
        "",
        "## Target leakage candidates (|r| ≥ 0.95)",
        "",
    ]
    for k, v in decisions["leakage_candidates"].items():
        lines.append(f"- **{k}**: r = {v} → {decisions['leakage_action'].get(k, '?')}")
    lines += [
        "",
        "## High feature-feature correlation (|r| ≥ 0.85)",
        pairs.to_markdown(index=False) if len(pairs) else "_none_",
        "",
        f"## NSM vs hour: r = {decisions['nsm_vs_hour_r']} → **{decisions['nsm_decision']}**",
        f"## Day_of_week vs WeekStatus: r = {decisions['dow_vs_weekstatus_r']} → **{decisions['dow_decision']}**",
        "",
        "## VIF (multicollinearity)",
        vif_df.to_markdown(index=False),
        "",
        "## Proposed DROP list for clean model inputs",
        "```",
        str(decisions["proposed_drop"]),
        "```",
        "",
        "## Proposed KEEP",
        f"- Numeric: {decisions['proposed_keep_numeric']}",
        f"- Categorical: {decisions['proposed_keep_categorical']}",
        "",
        "## Next",
        f"**{decisions['next_step']}**",
    ]
    (OUT / "DECISIONS.md").write_text("\n".join(lines), encoding="utf-8")

    print("Target correlation top 5:")
    print(corr_target.head())
    print(f"\nLeakage candidates: {list(decisions['leakage_candidates'].keys())}")
    print(f"Proposed DROP: {decisions['proposed_drop']}")
    print(f"\nWrote {OUT}/DECISIONS.md")


if __name__ == "__main__":
    main()
