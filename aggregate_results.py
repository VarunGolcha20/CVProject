import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "DEFEND", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "BUFFERING"]

def generate_global_matrices(benchmark_dir="benchmark"):
    base_path = Path(benchmark_dir)
    data = {
        "rule_based": {"y_true": [], "y_pred": []},
        "cnn":        {"y_true": [], "y_pred": []},
        "lstm":       {"y_true": [], "y_pred": []}
    }

    # Find all reports but filter out directories starting with "move_"
    all_reports = list(base_path.glob("**/failure_report.json"))
    report_files = [f for f in all_reports if "move_" not in f.parts[-2]]
    
    if not report_files:
        print("No valid failure_report.json files found.")
        return

    for report_path in report_files:
        with open(report_path, 'r') as f:
            report = json.load(f)
            cases = report.get("cases", [])
            for case in cases:
                gt = case.get("gt")
                preds = case.get("preds", {})
                if gt and gt in GESTURE_CLASSES:
                    for clf in data.keys():
                        p = preds.get(clf)
                        if p in GESTURE_CLASSES:
                            data[clf]["y_true"].append(gt)
                            data[clf]["y_pred"].append(p)

    # Plot and save
    for clf_name, values in data.items():
        if not values["y_true"]: continue
        cm = confusion_matrix(values["y_true"], values["y_pred"], labels=GESTURE_CLASSES)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GESTURE_CLASSES)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        ax.set_title(f"Global Results (Excluding Move Folders) - {clf_name.upper()}")
        plt.tight_layout()
        plt.savefig(base_path / f"global_confusion_{clf_name}_no_move.png")
        print(f"Saved: global_confusion_{clf_name}_no_move.png")

if __name__ == "__main__":
    generate_global_matrices()