"""Audit saved forecast-corpus bus balance using recorded electrochemical power."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
folder = root / "matlab_model_x650_v03" / "results" / "probability_dataset_v01"
manifest = pd.read_csv(folder / "scenario_manifest_v01.csv")[["scenario_id", "dcdc_efficiency_scale"]]
cols = ["scenario_id", "split", "bus_load_W", "fc_power_W", "battery_power_W",
        "unmet_power_W", "surplus_power_W", "bus_residual_W"]
data = pd.read_csv(folder / "thermal_trajectories_v01.csv", usecols=cols).merge(manifest, on="scenario_id")
actual_dc = np.minimum(data.fc_power_W * .94 * data.dcdc_efficiency_scale, 2000)
data["reconstructed_error_W"] = (data.bus_load_W + data.surplus_power_W
    - actual_dc - data.battery_power_W - data.unmet_power_W)
data["affected_step"] = data.reconstructed_error_W.abs() > 1
out = (data.groupby(["split", "scenario_id"], as_index=False)
       .agg(steps=("affected_step", "size"), affected_steps=("affected_step", "sum"),
            max_abs_error_W=("reconstructed_error_W", lambda x: x.abs().max())))
target = Path(__file__).resolve().parent / "R2_forecast_corpus_power_audit.csv"
out.to_csv(target, index=False)
print(f"rows={len(data)} affected_rows={data.affected_step.sum()} "
      f"affected_scenarios={(out.affected_steps > 0).sum()} "
      f"max_abs_W={data.reconstructed_error_W.abs().max():.3f} "
      f"saved_residual_max_W={data.bus_residual_W.abs().max():.3f}")
