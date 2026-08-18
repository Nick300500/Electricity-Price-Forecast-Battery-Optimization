"""Central configuration: default parameters and file locations.

Kept as plain module-level constants (not a class) so notebook cells can
import exactly the names they need, e.g. ``from battery_opt.config import
DEFAULT_STORAGE_PARAMS``.
"""

import os

# Battery specs, modeled roughly on a Sungrow SBR096, 3 modules.
DEFAULT_STORAGE_PARAMS = {
    "capacity": 9.6,    # kWh
    "p_max": 5.76,      # kW
    "eff_ch": 0.95,     # charging efficiency
    "eff_dis": 0.95,    # discharging efficiency
    "soc_init": 0,      # initial state of charge
}

# Assumed turnkey system cost used for the payback-period estimate.
BATTERY_SYSTEM_COST_EUR = 2930

RESULTS_DIR = "Results"
EXTRA_DIR = "Extra"
OUTPUTS_DIR = "outputs"
RF_MODEL_PATH = os.path.join(OUTPUTS_DIR, "rf_model.pkl")


def ensure_dirs():
    """Create the output directories used by the pipeline, if missing."""
    for d in (RESULTS_DIR, EXTRA_DIR, OUTPUTS_DIR):
        os.makedirs(d, exist_ok=True)
