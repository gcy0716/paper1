from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

with (RESULTS / "controller_comparison_v03.csv").open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 12, f"expected 4 controllers x 3 altitudes, got {len(rows)}"
assert {r["controller"] for r in rows} == {"fixed", "pi", "safe_pi", "mpc"}
assert {float(r["altitude_m"]) for r in rows} == {0.0, 3000.0, 6000.0}

for r in rows:
    assert abs(float(r["max_bus_residual_W"])) < 1e-8
    assert abs(float(r["max_thermal_residual_W"])) < 1e-8
    assert 0.25 - 1e-9 <= float(r["minimum_SOC"]) <= 0.95 + 1e-9
    assert float(r["hydrogen_used_g"]) > 0

pi0 = next(r for r in rows if r["controller"] == "pi" and float(r["altitude_m"]) == 0)
safe0 = next(r for r in rows if r["controller"] == "safe_pi" and float(r["altitude_m"]) == 0)
assert float(safe0["peak_stack_temperature_C"]) < float(pi0["peak_stack_temperature_C"])
assert float(safe0["safety_override_steps"]) > 0

h6000 = [r for r in rows if float(r["altitude_m"]) == 6000]
assert all(float(r["unmet_energy_kJ"]) > 0 for r in h6000)

report = RESULTS / "python_verification_v03.txt"
report.write_text(
    "V03_VERIFICATION_PASS\n"
    "12 controller-altitude cases present\n"
    "power and thermal residuals < 1e-8 W\n"
    "SOC bounds enforced\n"
    "safe PI reduces sea-level peak temperature relative to untuned PI\n"
    "6000 m energy shortage is detected, not hidden by SOC clipping\n",
    encoding="utf-8",
)
print(report.read_text(encoding="utf-8"))
