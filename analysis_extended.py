"""
analysis_extended.py
====================
Deep per-model, per-gesture error analysis for GestureBall.

Reads  : benchmark/detailed_log.json   (produced by benchmark.py)
Outputs: benchmark/analysis/           (7 PNG plots + console tables)

Usage:
    python analysis_extended.py
    python analysis_extended.py --log benchmark/detailed_log.json --out benchmark/analysis
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support, confusion_matrix,
)

# ── Constants ────────────────────────────────────────────────────────────────
GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "MOVE"]
MODEL_NAMES     = ["rule_based", "cnn", "lstm"]
MODEL_LABELS    = {"rule_based": "Rule-Based", "cnn": "CNN (PoseCNN)", "lstm": "LSTM/GRU"}
MODEL_COLORS    = {"rule_based": "#EF9F27",    "cnn": "#E85D30",       "lstm": "#3B8BD4"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":  150,
})


# ── Load ─────────────────────────────────────────────────────────────────────
def load_log(log_path: str) -> list:
    with open(log_path) as f:
        raw = json.load(f)
    labelled = [d for d in raw if d.get("gt") and d["gt"] in GESTURE_CLASSES]
    print(f"\nLoaded {len(raw)} total frames | {len(labelled)} with GT labels.")
    return labelled


# ── 1. Per-class F1 Heatmap ──────────────────────────────────────────────────
def plot_per_class_f1(data: list, out_dir: Path):
    models = [m for m in MODEL_NAMES if any(m in d["predictions"] for d in data)]
    f1_mat = np.zeros((len(models), len(GESTURE_CLASSES)))

    print("\n" + "─"*55)
    print("1. PER-CLASS F1 SCORES")
    print("─"*55)

    for i, model in enumerate(models):
        y_true = [d["gt"] for d in data if model in d["predictions"]]
        y_pred = [d["predictions"][model] for d in data if model in d["predictions"]]
        if not y_true:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=GESTURE_CLASSES, average=None, zero_division=0)
        f1_mat[i] = f1
        print(f"\n{MODEL_LABELS[model]}")
        print(classification_report(y_true, y_pred, labels=GESTURE_CLASSES, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(f1_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GESTURE_CLASSES))); ax.set_xticklabels(GESTURE_CLASSES, fontsize=12)
    ax.set_yticks(range(len(models)));          ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    for i in range(len(models)):
        for j in range(len(GESTURE_CLASSES)):
            v = f1_mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white" if v < 0.5 else "black")
    plt.colorbar(im, ax=ax, label="F1 Score", fraction=0.046)
    ax.set_title("Per-Class F1 Score by Model\n"
                 "(Red = low, Green = high — identifies each model's worst gesture)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    p = out_dir / "1_f1_heatmap.png"; plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")


# ── 2. Confidence Distributions ──────────────────────────────────────────────
def plot_confidence_distributions(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n2. CONFIDENCE DISTRIBUTIONS\n" + "─"*55)
    for model in ["cnn", "lstm"]:
        entries = [d for d in data
                   if model in d["predictions"] and
                   d.get("confidences", {}).get(model) is not None]
        if not entries:
            print(f"  [skip] No confidence data for {model}")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"Confidence Distributions — {MODEL_LABELS[model]}\n"
            "Green = correctly classified  |  Red = misclassified\n"
            "High-confidence errors → systematic model failure (not just boundary uncertainty)",
            fontsize=11, fontweight="bold")
        bins = np.linspace(0, 1, 21)

        for idx, gesture in enumerate(GESTURE_CLASSES):
            ax = axes[idx // 2][idx % 2]
            gframes = [d for d in entries if d["gt"] == gesture]
            correct = [d["confidences"][model] for d in gframes if d["predictions"][model] == gesture]
            wrong   = [d["confidences"][model] for d in gframes if d["predictions"][model] != gesture]

            if correct:
                ax.hist(correct, bins=bins, color="#27ae60", alpha=0.75,
                        label=f"Correct (n={len(correct)})", edgecolor="white")
            if wrong:
                ax.hist(wrong, bins=bins, color="#e74c3c", alpha=0.75,
                        label=f"Misclassified (n={len(wrong)})", edgecolor="white")
                if np.mean(wrong) > 0.65:
                    ax.text(0.03, 0.92, "High-confidence errors\n→ systematic failure",
                            transform=ax.transAxes, color="#e74c3c", fontsize=8, va="top")

            ax.set_title(f"GT = {gesture}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Max softmax confidence"); ax.set_ylabel("Frame count")
            ax.legend(fontsize=9); ax.set_xlim(0, 1)

        plt.tight_layout()
        p = out_dir / f"2_confidence_{model}.png"; plt.savefig(p, bbox_inches="tight"); plt.close()
        print(f"Saved: {p}")


# ── 3. GRU Attention Analysis ────────────────────────────────────────────────
def plot_attention_analysis(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n3. GRU ATTENTION ANALYSIS (PASS sequences)\n" + "─"*55)
    entries = [d for d in data
               if d.get("gt") == "PASS" and
               d.get("attention_weights") is not None and
               "lstm" in d.get("predictions", {})]
    if not entries:
        print("  [skip] No LSTM attention data found for PASS frames.")
        return

    correct = [d for d in entries if d["predictions"]["lstm"] == "PASS"]
    wrong   = [d for d in entries if d["predictions"]["lstm"] != "PASS"]
    n = 15
    groups  = [(correct[:n], "Correctly Classified PASS", "Greens"),
               (wrong[:n],   "PASS Misclassified",        "Reds")]

    ncols = sum(1 for g, _, _ in groups if g)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))
    if ncols == 1: axes = [axes]
    ax_idx = 0

    for group, title, cmap in groups:
        if not group: continue
        ax = axes[ax_idx]; ax_idx += 1
        mat = np.array([d["attention_weights"] for d in group])
        im  = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=mat.max())
        ax.set_xlabel("Frame index (0=oldest → N-1=newest)", fontsize=10)
        ax.set_ylabel("Sequence index", fontsize=10)
        ax.set_title(f"{title}\n(n={len(group)})", fontsize=11, fontweight="bold")
        ax.axvline(x=8,  color="yellow", linestyle="--", lw=1.5, label="Expected kick (~frame 8)")
        ax.legend(loc="upper left", fontsize=8)
        plt.colorbar(im, ax=ax, label="Attention weight")

        peaks = mat.argmax(axis=1)
        print(f"  {title}: mean peak attention frame = {peaks.mean():.1f} / {mat.shape[1]-1}")
        if peaks.mean() > mat.shape[1] * 0.6:
            print("  ⚠  Peak attention in last 40% of window → model attends to POST-kick recovery, not the kick")

    fig.suptitle("GRU Attention Weights for PASS Sequences\n"
                 "Each row = one 35-frame window. Bright cell = frame the model weighted most.",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = out_dir / "3_gru_attention_pass.png"; plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")


# ── 4. Rule-Based Decision Boundary Scatter ──────────────────────────────────
def plot_rule_based_boundary(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n4. RULE-BASED DECISION BOUNDARY\n" + "─"*55)
    rb_data = [d for d in data if "rule_based" in d.get("predictions", {})
               and d.get("features")]
    if not rb_data:
        print("  [skip] No rule-based data with features."); return

    colors = {"SHOOT": "#E85D30", "PASS": "#3B8BD4", "IDLE": "#888888", "MOVE": "#2ecc71"}
    markers = {"SHOOT": "^", "PASS": "o", "IDLE": "s", "MOVE": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Rule-Based Classifier: Decision Boundary Analysis\n"
                 "Threshold lines show where SHOOT/PASS rules fire. Points near lines → misclassification risk.",
                 fontsize=12, fontweight="bold")

    # --- Left: max of left/right forward ratio vs max lateral offset ---
    ax = axes[0]
    for gesture in GESTURE_CLASSES:
        pts = [d for d in rb_data if d["gt"] == gesture]
        if not pts: continue
        x = [max(d["features"]["left_forward_ratio"], d["features"]["right_forward_ratio"]) for d in pts]
        y = [max(abs(d["features"]["left_lateral_offset"]), abs(d["features"]["right_lateral_offset"])) for d in pts]
        wrong = [d["predictions"]["rule_based"] != gesture for d in pts]
        ax.scatter([xi for xi, w in zip(x, wrong) if not w],
                   [yi for yi, w in zip(y, wrong) if not w],
                   c=colors[gesture], marker=markers[gesture], alpha=0.5, s=30,
                   label=f"{gesture} (correct)")
        ax.scatter([xi for xi, w in zip(x, wrong) if w],
                   [yi for yi, w in zip(y, wrong) if w],
                   c=colors[gesture], marker="x", alpha=0.9, s=60, linewidths=1.5,
                   label=f"{gesture} (wrong)")

    ax.axvline(x=0.65, color="red",  linestyle="--", lw=1.5, label="SHOOT threshold (0.65)")
    ax.axhline(y=0.40, color="blue", linestyle="--", lw=1.5, label="PASS threshold (0.40)")
    ax.set_xlabel("Max foot forward ratio (→ SHOOT if >0.65)", fontsize=10)
    ax.set_ylabel("Max lateral offset  (→ PASS if >0.40)", fontsize=10)
    ax.set_title("SHOOT vs PASS Decision Boundary\n(× = misclassified)", fontsize=11)
    ax.legend(fontsize=7, ncol=2); ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 2.0)

    # --- Right: kinetic energy for IDLE frames, coloured by prediction ---
    ax2 = axes[1]
    idle_frames = [d for d in rb_data if d["gt"] == "IDLE"]
    if idle_frames:
        ke   = [d["features"]["total_kinetic_energy"] for d in idle_frames]
        mjc  = [d["features"]["moving_joint_count"]   for d in idle_frames]
        pred = [d["predictions"]["rule_based"]         for d in idle_frames]
        c_vals = ["#27ae60" if p == "IDLE" else "#e74c3c" for p in pred]
        ax2.scatter(ke, mjc, c=c_vals, alpha=0.65, s=35)
        ax2.set_xlabel("Total kinetic energy", fontsize=10)
        ax2.set_ylabel("Moving joint count (0–1)", fontsize=10)
        ax2.set_title("IDLE Frames: Kinetic Features\n(Green=correct, Red=misclassified as other)", fontsize=11)
        patch_c = mpatches.Patch(color="#27ae60", label="Classified IDLE (correct)")
        patch_w = mpatches.Patch(color="#e74c3c", label="Misclassified IDLE")
        ax2.legend(handles=[patch_c, patch_w], fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No IDLE frames in log", ha="center", transform=ax2.transAxes)

    plt.tight_layout()
    p = out_dir / "4_rule_boundary.png"; plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")

    # Summary stats
    shoot_as_pass = sum(1 for d in rb_data if d["gt"]=="SHOOT" and d["predictions"]["rule_based"]=="PASS")
    pass_as_shoot = sum(1 for d in rb_data if d["gt"]=="PASS"  and d["predictions"]["rule_based"]=="SHOOT")
    total_pass    = sum(1 for d in rb_data if d["gt"]=="PASS")
    total_shoot   = sum(1 for d in rb_data if d["gt"]=="SHOOT")
    print(f"  PASS→SHOOT errors: {pass_as_shoot}/{total_pass} ({100*pass_as_shoot/max(1,total_pass):.1f}%)")
    print(f"  SHOOT→PASS errors: {shoot_as_pass}/{total_shoot} ({100*shoot_as_pass/max(1,total_shoot):.1f}%)")


# ── 5. Cross-Model Error Correlation ─────────────────────────────────────────
def plot_cross_model_errors(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n5. CROSS-MODEL ERROR CORRELATION\n" + "─"*55)
    models = [m for m in MODEL_NAMES if any(m in d["predictions"] for d in data)]
    if len(models) < 2:
        print("  [skip] Need at least 2 models."); return

    wrong = {m: {d["frame"]: d["predictions"][m] != d["gt"]
                 for d in data if m in d["predictions"]} for m in models}
    frames = list({d["frame"] for d in data})

    categories, counts = [], []
    all_w = [i for i in frames if all(wrong[m].get(i, False) for m in models)]
    categories.append("All models wrong\n(lighting / occlusion)"); counts.append(len(all_w))
    for m in models:
        only_m = [i for i in frames if wrong[m].get(i, False) and
                  all(not wrong[m2].get(i, False) for m2 in models if m2 != m)]
        categories.append(f"Only {MODEL_LABELS[m]}\nwrong"); counts.append(len(only_m))
        print(f"  Only {MODEL_LABELS[m]} wrong: {len(only_m)} frames")
    print(f"  All models wrong:  {len(all_w)} frames → shared failure (pose/lighting)")

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#95a5a6"] + [MODEL_COLORS[m] for m in models]
    bars = ax.bar(categories, counts, color=bar_colors, width=0.55, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of frames", fontsize=11)
    ax.set_title("Cross-Model Error Correlation\n"
                 "Shared failures → fundamental ambiguity; Model-only failures → fixable limitations",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / "5_cross_model_errors.png"; plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")


# ── 6. Error Rate vs Visibility ──────────────────────────────────────────────
def plot_visibility_vs_error(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n6. ERROR RATE VS VISIBILITY\n" + "─"*55)
    bins  = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    labels = ["0.0–0.3\n(poor)", "0.3–0.5\n(low)", "0.5–0.7\n(medium)", "0.7–1.0\n(good)"]
    models = [m for m in MODEL_NAMES if any(m in d["predictions"] for d in data)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(bins))
    width = 0.22
    offsets = np.linspace(-(len(models)-1)*width/2, (len(models)-1)*width/2, len(models))

    for model, offset in zip(models, offsets):
        error_rates = []
        for lo, hi in bins:
            bucket = [d for d in data if model in d["predictions"]
                      and lo <= d["visibility"] < hi]
            if not bucket:
                error_rates.append(0); continue
            err = sum(1 for d in bucket if d["predictions"][model] != d["gt"]) / len(bucket)
            error_rates.append(err * 100)
        bars = ax.bar(x + offset, error_rates, width, label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")
        print(f"  {MODEL_LABELS[model]}: {[f'{e:.1f}%' for e in error_rates]}")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean Visibility (MediaPipe landmark confidence)", fontsize=11)
    ax.set_ylabel("Error rate (%)", fontsize=11)
    ax.set_title("Error Rate vs Landmark Visibility per Model\n"
                 "All models degrade in poor lighting; LSTM degrades least (temporal averaging helps)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout()
    p = out_dir / "6_visibility_error.png"; plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")


# ── 7. Per-Gesture Error Rate (grouped bar) ───────────────────────────────────
def plot_per_gesture_error_rate(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n7. PER-GESTURE ERROR RATE BY MODEL\n" + "─"*55)
    models = [m for m in MODEL_NAMES if any(m in d["predictions"] for d in data)]
    x      = np.arange(len(GESTURE_CLASSES))
    width  = 0.22
    offsets = np.linspace(-(len(models)-1)*width/2, (len(models)-1)*width/2, len(models))

    fig, ax = plt.subplots(figsize=(11, 5))

    for model, offset in zip(models, offsets):
        err_rates = []
        for gesture in GESTURE_CLASSES:
            frames = [d for d in data if d["gt"] == gesture and model in d["predictions"]]
            if not frames:
                err_rates.append(0); continue
            err = sum(1 for d in frames if d["predictions"][model] != gesture) / len(frames)
            err_rates.append(err * 100)
        bars = ax.bar(x + offset, err_rates, width, label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, err_rates):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
        print(f"  {MODEL_LABELS[model]}: " +
              " | ".join(f"{g}={e:.1f}%" for g, e in zip(GESTURE_CLASSES, err_rates)))

    ax.set_xticks(x); ax.set_xticklabels(GESTURE_CLASSES, fontsize=12)
    ax.set_xlabel("Ground Truth Gesture", fontsize=11)
    ax.set_ylabel("Error rate (%)", fontsize=11)
    ax.set_title("Per-Gesture Error Rate by Model\n"
                 "Identifies which gesture each model struggles with most",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout()
    p = out_dir / "7_per_gesture_error.png"; plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_analysis(log_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_log(log_path)
    if not data:
        print("No labelled frames found in log. Run benchmark.py with a GT labels CSV first.")
        return

    plot_per_class_f1(data, out)
    plot_confidence_distributions(data, out)
    plot_attention_analysis(data, out)
    plot_rule_based_boundary(data, out)
    plot_cross_model_errors(data, out)
    plot_visibility_vs_error(data, out)
    plot_per_gesture_error_rate(data, out)

    print(f"\n{'='*55}")
    print(f"Analysis complete. All plots saved to: {out}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GestureBall Extended Analysis")
    parser.add_argument("--log", default="benchmark/detailed_log.json",
                        help="Path to detailed_log.json from benchmark.py")
    parser.add_argument("--out", default="benchmark/analysis",
                        help="Output directory for plots")
    args = parser.parse_args()
    run_analysis(args.log, args.out)
