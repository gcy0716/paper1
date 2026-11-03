from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "evidence" / "figure6_digitized_v03.csv").open(encoding="utf-8-sig")))
current = np.array([float(r["current_A"]) for r in rows])
voltage = np.array([float(r["stack_voltage_V"]) for r in rows])

n_cells = 50.0
i_limit = 42.0
concentration = 0.03  # constrained positive assumption
transport = -np.log(1.0 - current / i_limit)

# Linear least squares for V/N = Voc - a*ln(1+I) - R*I - b*transport.
y = voltage / n_cells + concentration * transport
x = np.column_stack([np.ones_like(current), -np.log1p(current), -current])
voc, activation, ohmic = np.linalg.lstsq(x, y, rcond=None)[0]
prediction = n_cells * (x @ np.array([voc, activation, ohmic]) - concentration * transport)
rmse = float(np.sqrt(np.mean((prediction - voltage) ** 2)))

assert voc > 0 and activation > 0 and ohmic > 0
assert rmse < 0.30

out = ROOT / "results" / "polarization_calibration_v03.txt"
out.write_text(
    f"Voc_cell_V={voc:.8f}\n"
    f"activation_V={activation:.8f}\n"
    f"ohmic_Ohm={ohmic:.8f}\n"
    f"concentration_V={concentration:.8f}\n"
    f"voltage_RMSE_V={rmse:.6f}\n"
    "status=PASS_WITHIN_DIGITIZATION_UNCERTAINTY\n",
    encoding="utf-8",
)
print(out.read_text(encoding="utf-8"))
