from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "probability_dataset_v01" / "quantile_gru_results"
OUT_DIR = ROOT / "results" / "probability_dataset_v01" / "figures"
OUT_DIR.mkdir(exist_ok=True)

metrics = pd.read_csv(DATA_DIR / "quantile_gru_calibrated_metrics_v01.csv")
calibration = pd.read_csv(DATA_DIR / "quantile_gru_calibration_v01.csv")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 600,
    }
)

colors = {"test": "#0077BB", "ood_test": "#EE7733"}
labels = {"test": "Test", "ood_test": "OOD test"}
markers = {"raw": "o", "validation_conformal": "s"}
variant_labels = {"raw": "Raw", "validation_conformal": "Calibrated"}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))

# (a) Point-prediction error is unaffected by interval calibration.
for split in ("test", "ood_test"):
    d = metrics[(metrics.split == split) & (metrics.variant == "raw")]
    axes[0].plot(d.horizon_s, d.mae_C, marker="o", lw=1.5, color=colors[split], label=labels[split])
axes[0].set(xlabel="Prediction horizon (s)", ylabel="MAE (°C)", xticks=[10, 30, 60])
axes[0].set_title("(a) Point accuracy")

# (b) P10-P90 empirical coverage before and after validation-only calibration.
for split in ("test", "ood_test"):
    for variant in ("raw", "validation_conformal"):
        d = metrics[(metrics.split == split) & (metrics.variant == variant)]
        axes[1].plot(d.horizon_s, 100 * d.picp_p10_p90, marker=markers[variant], lw=1.4,
                     ls="-" if variant == "raw" else "--", color=colors[split],
                     label=f"{labels[split]}, {variant_labels[variant]}")
axes[1].axhline(80, color="#555555", ls=":", lw=1.1, label="Nominal 80%")
axes[1].set(xlabel="Prediction horizon (s)", ylabel="P10–P90 coverage (%)", xticks=[10, 30, 60], ylim=(0, 105))
axes[1].set_title("(b) Interval calibration")

# (c) Safety-relevant exceedance recall after calibration.
for split in ("test", "ood_test"):
    d = metrics[(metrics.split == split) & (metrics.variant == "validation_conformal")]
    axes[2].plot(d.horizon_s, 100 * d.overtemp_recall, marker="o", lw=1.5,
                 color=colors[split], label=labels[split])
axes[2].set(xlabel="Prediction horizon (s)", ylabel="Overtemperature recall (%)", xticks=[10, 30, 60], ylim=(0, 105))
axes[2].set_title("(c) Risk detection")

for ax in axes:
    ax.grid(alpha=0.2)

legend_handles = [
    Line2D([0], [0], color=colors["test"], marker="o", lw=1.5, label="Test domain"),
    Line2D([0], [0], color=colors["ood_test"], marker="o", lw=1.5, label="OOD domain"),
    Line2D([0], [0], color="#555555", marker="o", lw=1.4, ls="-", label="Raw interval"),
    Line2D([0], [0], color="#555555", marker="s", lw=1.4, ls="--", label="Calibrated interval"),
    Line2D([0], [0], color="#555555", lw=1.1, ls=":", label="Nominal 80%"),
]
fig.legend(handles=legend_handles, ncol=5, frameon=False, fontsize=7,
           loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.88))

for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"Fig_probability_calibration_v02.{ext}", bbox_inches="tight")
plt.close(fig)

calibration.to_csv(OUT_DIR / "probability_calibration_source_v02.csv", index=False)
print(f"Saved probability calibration figure to {OUT_DIR}")
