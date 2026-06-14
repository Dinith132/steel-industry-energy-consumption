"""Run comprehensive Final-results analysis. Used by findings_summary.ipynb logic."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- config ---
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parent / "Final-results" / "Final"
FINDINGS_DIR = RESULTS_ROOT / "findings"
PLOTS_DIR = FINDINGS_DIR / "plots"

TRACKS = ["hourly", "15min"]
STACKS = ["single", "double", "bidir"]
WINDOWS = {
    "hourly": [1, 4, 8, 12, 16, 24, 36, 48, 74, 168, 336, 672],
    "15min": [1, 4, 8, 16, 24, 48, 64, 96, 672],
}
COMPARE_WINDOW = {"hourly": 24, "15min": 96}
BEHAVIOR_DIRS = ["shap", "ig", "erasure", "fidelity"]


def _parse_model_path(rel_path: str) -> dict:
    """Extract track/stack/window from behaviors paths."""
    info = {"track": None, "stack": None, "window": None, "category": None}
    parts = Path(rel_path).parts
    for t in TRACKS:
        if t in parts:
            info["track"] = t
            break
    for s in STACKS:
        if s in parts:
            info["stack"] = s
            break
    for cat in BEHAVIOR_DIRS + ["xai", "ig", "analysis", "plots", "models", "history"]:
        if cat in parts:
            info["category"] = cat
            break
    m = re.search(r"win(\d+)", rel_path)
    if m:
        info["window"] = int(m.group(1))
    return info


def part0_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fp in sorted(RESULTS_ROOT.rglob("*")):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(RESULTS_ROOT))
        meta = _parse_model_path(rel)
        rows.append({
            "path": rel,
            "ext": fp.suffix.lower(),
            "size_bytes": fp.stat().st_size,
            **meta,
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(FINDINGS_DIR / "file_manifest.csv", index=False)

    gaps = []
    for track in TRACKS:
        for stack in STACKS:
            for window in WINDOWS[track]:
                for beh in BEHAVIOR_DIRS:
                    csv_p = RESULTS_ROOT / track / "behaviors" / beh / stack / f"win{window}.csv"
                    if not csv_p.exists():
                        gaps.append({
                            "track": track, "stack": stack, "window": window,
                            "missing": f"behaviors/{beh}/{stack}/win{window}.csv",
                        })
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(FINDINGS_DIR / "gaps_report.csv", index=False)
    return manifest, gaps_df


def load_metrics() -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for fname in ["results_metrics.csv", "analysis/recomputed_metrics.csv"]:
            p = RESULTS_ROOT / track / fname
            if p.exists():
                df = pd.read_csv(p)
                df["track"] = track
                df["source"] = fname
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_shap_all() -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for stack in STACKS:
            for window in WINDOWS[track]:
                p = RESULTS_ROOT / track / "behaviors" / "shap" / stack / f"win{window}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                top = df.sort_values("value", ascending=False)
                rows.append({
                    "track": track, "stack": stack, "window": window,
                    "top1_feature": top.iloc[0]["attr"],
                    "top1_value": float(top.iloc[0]["value"]),
                    "top2_feature": top.iloc[1]["attr"] if len(top) > 1 else None,
                    "top3_feature": top.iloc[2]["attr"] if len(top) > 2 else None,
                    "usage_kwh_rank1": top.iloc[0]["attr"] == "Usage_kWh",
                })
    return pd.DataFrame(rows)


def ig_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    col = "mean_abs_attr" if "mean_abs_attr" in df.columns else df.columns[-1]
    v = df[col].to_numpy(dtype=float)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return {k: np.nan for k in [
            "recency_ratio", "start_ratio", "mid_ratio", "u_shape_score", "peak_step", "ig_pattern"
        ]}
    q1 = max(1, n // 4)
    q3 = min(n, max(q1 + 1, 3 * n // 4))
    start = v[:q1].sum()
    mid = v[q1:q3].sum() if q3 > q1 else 0.0
    end = v[q3:].sum() if q3 < n else 0.0
    total = v.sum()
    recency = end / total
    start_r = start / total
    mid_r = mid / total
    peak = int(np.argmax(v))
    u_score = (start + end) / max(mid, total * 0.02)

    if recency >= 0.55 and u_score < 3.0:
        pattern = "recency_dominant"
    elif u_score >= 3.0 and start_r >= 0.06:
        pattern = "u_shaped"
    elif max(v) / max(v.mean(), 1e-12) < 2.0:
        pattern = "flat"
    else:
        pattern = "mixed"

    return {
        "recency_ratio": recency,
        "start_ratio": start_r,
        "mid_ratio": mid_r,
        "u_shape_score": u_score,
        "peak_step": peak,
        "ig_pattern": pattern,
    }


def load_ig_all() -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for stack in STACKS:
            for window in WINDOWS[track]:
                p = RESULTS_ROOT / track / "behaviors" / "ig" / stack / f"win{window}.csv"
                if not p.exists():
                    continue
                m = ig_metrics(p)
                rows.append({"track": track, "stack": stack, "window": window, **m})
    return pd.DataFrame(rows)


def load_erasure_all() -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for stack in STACKS:
            for window in WINDOWS[track]:
                p = RESULTS_ROOT / track / "behaviors" / "erasure" / stack / f"win{window}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                base = df.loc[df["cutoff"] == 0.0, "rmse_kwh"].iloc[0]
                d25 = df.loc[df["cutoff"] == 0.25, "rmse_kwh"].iloc[0] - base
                d50 = df.loc[df["cutoff"] == 0.5, "rmse_kwh"].iloc[0] - base
                d75 = df.loc[df["cutoff"] == 0.75, "rmse_kwh"].iloc[0] - base
                rows.append({
                    "track": track, "stack": stack, "window": window,
                    "erasure_baseline_rmse": base,
                    "delta_25": d25, "delta_50": d50, "delta_75": d75,
                    "total_sensitivity": df["rmse_kwh"].iloc[-1] - base,
                })
    return pd.DataFrame(rows)


def load_fidelity_all() -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for stack in STACKS:
            for window in WINDOWS[track]:
                p = RESULTS_ROOT / track / "behaviors" / "fidelity" / stack / f"win{window}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                idx = df["delta"].idxmax()
                rows.append({
                    "track": track, "stack": stack, "window": window,
                    "max_delta": float(df["delta"].max()),
                    "top_faithful_feature": df.loc[idx, "attr"],
                    "usage_kwh_max_delta": float(
                        df.loc[df["attr"] == "Usage_kWh", "delta"].max()
                        if (df["attr"] == "Usage_kWh").any() else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def load_lime_spearman() -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        xai_dir = RESULTS_ROOT / track / "analysis" / "xai"
        if not xai_dir.exists():
            continue
        for fp in xai_dir.rglob("*_shap_lime_corr.csv"):
            df = pd.read_csv(fp)
            if len(df) == 0:
                continue
            row = df.iloc[0].to_dict()
            row["track"] = track
            rows.append(row)
    return pd.DataFrame(rows)


def build_master(
    perf: pd.DataFrame, shap: pd.DataFrame, ig: pd.DataFrame,
    erasure: pd.DataFrame, fidelity: pd.DataFrame,
) -> pd.DataFrame:
    perf_main = perf[perf["source"] == "results_metrics.csv"].copy()
    perf_main = perf_main.rename(columns={"model": "stack"})
    perf_main = perf_main[["track", "stack", "window", "rmse_kwh", "mae_kwh", "r2", "wia"]]

    master = perf_main
    for df in [shap, ig, erasure, fidelity]:
        master = master.merge(df, on=["track", "stack", "window"], how="left")

    best_path = RESULTS_ROOT / "analysis" / "best_models.json"
    if best_path.exists():
        with open(best_path) as f:
            best = json.load(f)
        master["is_best_track"] = master.apply(
            lambda r: (
                r["track"] in best
                and r["stack"] == best[r["track"]]["model"]
                and r["window"] == best[r["track"]]["window"]
            ),
            axis=1,
        )
    return master


def plot_rmse_heatmap(master: pd.DataFrame, track: str, out: Path):
    sub = master[master["track"] == track]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="stack", columns="window", values="rmse_kwh")
    plt.figure(figsize=(12, 3))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd_r")
    plt.title(f"{track} — RMSE (kWh) by architecture and window")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_recency_heatmap(master: pd.DataFrame, track: str, out: Path):
    sub = master[master["track"] == track]
    pivot = sub.pivot_table(index="stack", columns="window", values="recency_ratio")
    plt.figure(figsize=(12, 3))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"{track} — IG recency ratio (last 25% / total)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_ig_compare(track: str, window: int, out: Path):
    plt.figure(figsize=(9, 4))
    for stack in STACKS:
        p = RESULTS_ROOT / track / "behaviors" / "ig" / stack / f"win{window}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        col = "mean_abs_attr" if "mean_abs_attr" in d.columns else d.columns[-1]
        plt.plot(d["step"] if "step" in d.columns else range(len(d)), d[col], "o-", label=stack)
    plt.xlabel("Time step (0 = oldest, right = recent)")
    plt.ylabel("Mean |attribution|")
    plt.title(f"{track} — IG memory horizon at window {window}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_all_figures(master: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        plot_rmse_heatmap(master, track, PLOTS_DIR / f"rmse_heatmap_{track}.png")
        plot_recency_heatmap(master, track, PLOTS_DIR / f"recency_heatmap_{track}.png")
        plot_ig_compare(track, COMPARE_WINDOW[track], PLOTS_DIR / f"ig_compare_win{COMPARE_WINDOW[track]}_{track}.png")

    # recency by stack boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=master, x="stack", y="recency_ratio", hue="track")
    plt.title("Recency ratio by architecture (all models)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recency_ratio_by_stack.png", dpi=150)
    plt.close()

    # recency vs window lines
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, track in zip(axes, TRACKS):
        sub = master[master["track"] == track]
        for stack in STACKS:
            s = sub[sub["stack"] == stack].sort_values("window")
            ax.plot(s["window"], s["recency_ratio"], "o-", label=stack)
        ax.set_xscale("log")
        ax.set_xlabel("Window size")
        ax.set_ylabel("Recency ratio")
        ax.set_title(track)
        ax.legend()
    plt.suptitle("Recency ratio vs window size")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recency_vs_window.png", dpi=150)
    plt.close()

    # SHAP top1 frequency
    usage_pct = master.groupby(["track", "stack"])["usage_kwh_rank1"].mean().reset_index()
    usage_pct["pct"] = usage_pct["usage_kwh_rank1"] * 100
    plt.figure(figsize=(8, 4))
    sns.barplot(data=usage_pct, x="stack", y="pct", hue="track")
    plt.ylabel("% models with Usage_kWh as top SHAP feature")
    plt.title("Autoregressive feature dominance (SHAP)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_usage_kwh_top1_pct.png", dpi=150)
    plt.close()

    # erasure delta heatmap hourly
    for track in TRACKS:
        sub = master[master["track"] == track]
        pivot = sub.pivot_table(index="stack", columns="window", values="delta_25")
        plt.figure(figsize=(12, 3))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds")
        plt.title(f"{track} — RMSE increase when oldest 25% erased")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"erasure_delta25_heatmap_{track}.png", dpi=150)
        plt.close()

    # mean erasure curve by stack
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, track in zip(axes, TRACKS):
        for stack in STACKS:
            curves = []
            for window in WINDOWS[track]:
                p = RESULTS_ROOT / track / "behaviors" / "erasure" / stack / f"win{window}.csv"
                if p.exists():
                    d = pd.read_csv(p)
                    base = d.loc[d["cutoff"] == 0.0, "rmse_kwh"].iloc[0]
                    curves.append((d["cutoff"], d["rmse_kwh"] - base))
            if not curves:
                continue
            cutoffs = curves[0][0]
            mean_delta = np.mean([c[1].values for c in curves], axis=0)
            ax.plot(cutoffs, mean_delta, "o-", label=stack)
        ax.set_xlabel("Oldest fraction erased")
        ax.set_ylabel("Mean RMSE increase (kWh)")
        ax.set_title(track)
        ax.legend()
    plt.suptitle("Memory erasure — average across all windows")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "erasure_mean_curves.png", dpi=150)
    plt.close()

    # IG vs erasure scatter
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=master, x="recency_ratio", y="delta_25", hue="stack", style="track", s=80)
    plt.xlabel("IG recency ratio")
    plt.ylabel("RMSE delta at 25% erasure")
    plt.title("IG–erasure agreement (all models)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ig_erasure_scatter.png", dpi=150)
    plt.close()

    # fidelity heatmap
    for track in TRACKS:
        sub = master[master["track"] == track]
        pivot = sub.pivot_table(index="stack", columns="window", values="max_delta")
        plt.figure(figsize=(12, 3))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Oranges")
        plt.title(f"{track} — max fidelity delta (zero top SHAP feature)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"fidelity_heatmap_{track}.png", dpi=150)
        plt.close()

    # RMSE vs recency
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=master, x="recency_ratio", y="rmse_kwh", hue="stack", style="track", s=80)
    plt.xlabel("Recency ratio")
    plt.ylabel("RMSE (kWh)")
    plt.title("Prediction accuracy vs memory recency bias")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rmse_vs_recency.png", dpi=150)
    plt.close()

    # u_shape by stack
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=master, x="stack", y="u_shape_score", hue="track")
    plt.title("U-shape score by architecture (bidir boundary sensitivity)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "u_shape_by_stack.png", dpi=150)
    plt.close()

    # IG facet key windows hourly
    key_wins = [8, 24, 168]
    fig, axes = plt.subplots(len(key_wins), 1, figsize=(9, 10))
    for ax, w in zip(axes, key_wins):
        for stack in STACKS:
            p = RESULTS_ROOT / "hourly" / "behaviors" / "ig" / stack / f"win{w}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            col = "mean_abs_attr" if "mean_abs_attr" in d.columns else d.columns[-1]
            ax.plot(d["step"] if "step" in d.columns else range(len(d)), d[col], "o-", label=stack)
        ax.set_title(f"hourly window {w}")
        ax.set_xlabel("step (0=oldest)")
        ax.legend()
    plt.suptitle("IG curves — key windows (hourly)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ig_facet_key_windows_hourly.png", dpi=150)
    plt.close()

    # pattern counts
    pat = master.groupby(["track", "stack", "ig_pattern"]).size().reset_index(name="count")
    plt.figure(figsize=(10, 4))
    sns.barplot(data=pat, x="stack", y="count", hue="ig_pattern")
    plt.title("IG pattern classification counts")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ig_pattern_counts.png", dpi=150)
    plt.close()


def write_findings_md(master: pd.DataFrame, manifest: pd.DataFrame, gaps: pd.DataFrame) -> str:
    n_files = len(manifest)
    n_models = len(master)
    best_h = master[(master["track"] == "hourly") & master.get("is_best_track", False)].iloc[0] if master.get("is_best_track") is not None and master["is_best_track"].any() else None
    best_15 = master[(master["track"] == "15min") & master.get("is_best_track", False)].iloc[0] if master.get("is_best_track") is not None and master["is_best_track"].any() else None

    def pct_recency(sub):
        return 100 * (sub["recency_ratio"] > 0.5).mean() if len(sub) else 0

    sd = master[master["stack"].isin(["single", "double"])]
    bi = master[master["stack"] == "bidir"]

    lines = [
        "# Research Findings — LSTM Behavior Audit",
        "",
        f"Generated from **{n_files}** files across hourly and 15min tracks ({n_models} trained models with full behavior outputs).",
        "",
        "## Executive summary",
        "",
        "We audited 63 LSTM models (single, double, bidirectional) using SHAP, Integrated Gradients,",
        "memory erasure, and fidelity tests on UCI steel plant energy data.",
        "",
        "## Inventory",
        "",
        f"- Total files catalogued: **{n_files}**",
        f"- Missing behavior CSVs: **{len(gaps)}**",
        f"- Models in master table: **{n_models}**",
        "",
        "## Numbered findings",
        "",
    ]

    # Finding 1
    p_sd = pct_recency(sd)
    lines += [
        "### 1. Recency bias in single and double LSTM",
        f"- {p_sd:.0f}% of single/double models have recency_ratio > 0.5 (IG mass in last 25% of window).",
        f"- Mean recency_ratio single/double: **{sd['recency_ratio'].mean():.3f}** vs bidir: **{bi['recency_ratio'].mean():.3f}**.",
        "- IG curves typically flat on old timesteps with spike at the most recent step.",
        "",
    ]

    # Finding 2
    lines += [
        "### 2. Bidirectional boundary sensitivity (U-shape)",
        f"- Mean u_shape_score: bidir **{bi['u_shape_score'].mean():.2f}** vs single **{master[master['stack']=='single']['u_shape_score'].mean():.2f}**.",
        f"- Mean start_ratio (oldest 25%): bidir **{bi['start_ratio'].mean():.3f}** vs single **{master[master['stack']=='single']['start_ratio'].mean():.3f}**.",
        f"- {(bi['ig_pattern']=='u_shaped').mean()*100:.0f}% of bidir models classified as u_shaped vs {(master[master['stack']=='single']['ig_pattern']=='u_shaped').mean()*100:.0f}% single.",
        "- BiLSTM attributes importance to window start AND end; middle timesteps contribute less.",
        "",
    ]

    # Finding 3
    long_w = master[master["window"] >= 168]
    short_w = master[master["window"] <= 8]
    lines += [
        "### 3. Long window does not mean long memory",
        f"- Mean recency_ratio window>=168: **{long_w['recency_ratio'].mean():.3f}** vs window<=8: **{short_w['recency_ratio'].mean():.3f}**.",
        "- Larger input windows do not spread IG attribution across more history.",
        "",
    ]

    # Finding 4
    usage_pct = master["usage_kwh_rank1"].mean() * 100
    lines += [
        "### 4. Autoregressive feature dominance",
        f"- **{usage_pct:.0f}%** of all models rank Usage_kWh as #1 SHAP feature.",
        f"- Mean max fidelity delta when zeroing top features: **{master['max_delta'].mean():.2f}** kWh RMSE increase.",
        "",
    ]

    # Finding 5
    if best_h is not None:
        lines += [
            "### 5. Best prediction accuracy",
            f"- Hourly best: **{best_h['stack']}** window **{int(best_h['window'])}** RMSE **{best_h['rmse_kwh']:.2f}** kWh.",
        ]
    if best_15 is not None:
        lines += [
            f"- 15min best: **{best_15['stack']}** window **{int(best_15['window'])}** RMSE **{best_15['rmse_kwh']:.2f}** kWh.",
        ]
    lines += [
        "- Lower RMSE does not require lower recency_ratio (accuracy and memory depth differ).",
        "",
    ]

    # Finding 6
    corr = master["recency_ratio"].corr(master["delta_25"])
    lines += [
        "### 6. IG–erasure consistency",
        f"- Correlation(recency_ratio, erasure delta@25%): **{corr:.3f}**.",
        "- Erasure removes *oldest* steps; high recency models may still show moderate delta when old data is erased.",
        "- Use IG + erasure together: IG shows *where* importance lies; erasure shows *causal* impact of removing history.",
        "",
    ]

    # Finding 7
    lines += [
        "### 7. Hourly vs 15-minute resolution",
        f"- Hourly mean recency_ratio: **{master[master['track']=='hourly']['recency_ratio'].mean():.3f}**.",
        f"- 15min mean recency_ratio: **{master[master['track']=='15min']['recency_ratio'].mean():.3f}**.",
        "",
        "## Limitations",
        "",
        "- Kernel SHAP is noisy on very long windows (672).",
        "- LIME/Spearman available only on analysis/ subset, not all 63 models.",
        "- Fidelity uses zero in scaled space, not mean imputation.",
        "- IG step 0 = oldest timestep; right side = most recent.",
        "",
        "## Figure index",
        "",
    ]
    for fp in sorted(PLOTS_DIR.glob("*.png")):
        lines.append(f"- `plots/{fp.name}`")
    lines.append("")
    lines.append("## Data files")
    lines.append("- `master_metrics.csv` — one row per model, all derived metrics")
    lines.append("- `file_manifest.csv` — full file inventory")
    lines.append("- `gaps_report.csv` — missing expected behavior outputs")

    text = "\n".join(lines)
    (FINDINGS_DIR / "FINDINGS.md").write_text(text, encoding="utf-8")
    return text


def main(results_root: Path | None = None):
    global RESULTS_ROOT, FINDINGS_DIR, PLOTS_DIR
    if results_root is not None:
        RESULTS_ROOT = Path(results_root)
        FINDINGS_DIR = RESULTS_ROOT / "findings"
        PLOTS_DIR = FINDINGS_DIR / "plots"
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Part 0: inventory...")
    manifest, gaps = part0_inventory()
    print(f"  files: {len(manifest)}, gaps: {len(gaps)}")

    print("Parts 1-5: load tables...")
    perf = load_metrics()
    shap = load_shap_all()
    ig = load_ig_all()
    erasure = load_erasure_all()
    fidelity = load_fidelity_all()
    lime = load_lime_spearman()
    if len(lime) > 0:
        lime.to_csv(FINDINGS_DIR / "lime_spearman_subset.csv", index=False)

    master = build_master(perf, shap, ig, erasure, fidelity)
    master.to_csv(FINDINGS_DIR / "master_metrics.csv", index=False)
    print(f"  master rows: {len(master)}")

    print("Part 6: plots...")
    plot_all_figures(master)

    print("Part 7: FINDINGS.md...")
    write_findings_md(master, manifest, gaps)
    print(f"Done. Output: {FINDINGS_DIR}")


if __name__ == "__main__":
    main()
