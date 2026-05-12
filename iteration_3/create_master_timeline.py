import json
from pathlib import Path

import pandas as pd
from scipy.stats import qmc

# Configuration
STEP_SIZE = round((1.0 / 30), 2)
OUTPUT_FILE = "master_timeline.csv"
SIMULATION_CONFIG_FILE = Path(__file__).resolve().parent / "simulation_config.json"

with open(SIMULATION_CONFIG_FILE, "r") as f:
    _sim_config = json.load(f)

CONTROLS = {
    name: {
        "min": cfg["min"],
        "max": cfg["max"],
        "baseline": _sim_config["default_state"][name]["value"],
    }
    for name, cfg in _sim_config["controls"].items()
}

control_names = list(CONTROLS.keys())

# State tracking
current_time = 0.0
current_state = {name: cfg["baseline"] for name, cfg in CONTROLS.items()}

time_history = []
state_history = {name: [] for name in control_names}


def append_state(state, t):
    time_history.append(t)
    for name in control_names:
        state_history[name].append(state[name])


def hold(duration):
    global current_time
    steps = int(duration / STEP_SIZE)
    for _ in range(steps):
        current_time += STEP_SIZE
        append_state(current_state, round(current_time, 2))


def transition_to(target_state, duration):
    global current_time, current_state
    steps = int(duration / STEP_SIZE)

    start_state = current_state.copy()

    for i in range(1, steps + 1):
        current_time += STEP_SIZE
        fraction = i / float(steps)
        for name in control_names:
            current_state[name] = (
                start_state[name] + (target_state[name] - start_state[name]) * fraction
            )
        append_state(current_state, current_time)


# Phase 0: Baseline Burn-in
hold(600)

# Phase 1: Isolated Single Sweeps
for var in control_names:
    target = current_state.copy()

    # Sweep up
    target[var] = CONTROLS[var]["max"]
    transition_to(target, 15)
    hold(90)

    # Sweep down
    target[var] = CONTROLS[var]["min"]
    transition_to(target, 30)
    hold(90)

    # Return to baseline
    target[var] = CONTROLS[var]["baseline"]
    transition_to(target, 15)
    hold(60)

# Phase 2: Paired Sweeps (Example: Compliance down, Shunt up)
target = current_state.copy()
target["TotalCompliance"] = CONTROLS["TotalCompliance"]["min"]
target["cShuntFrac"] = CONTROLS["cShuntFrac"]["max"]
transition_to(target, 20)
hold(90)

# Return to baseline
target["TotalCompliance"] = CONTROLS["TotalCompliance"]["baseline"]
target["cShuntFrac"] = CONTROLS["cShuntFrac"]["baseline"]
transition_to(target, 20)
hold(60)

# Phase 3: Dynamic Stepping with Sobol Sequence
NUM_SOBOL_POINTS = 128
sampler = qmc.Sobol(d=len(control_names), scramble=True)
sobol_samples = sampler.random(n=NUM_SOBOL_POINTS)

# Scale Sobol samples [0, 1] to parameter bounds
for sample in sobol_samples:
    target = {}
    for i, name in enumerate(control_names):
        val_min = CONTROLS[name]["min"]
        val_max = CONTROLS[name]["max"]
        target[name] = val_min + sample[i] * (val_max - val_min)

    transition_to(target, 15)
    hold(90)

# Export to CSV
df = pd.DataFrame({"time": time_history})
for name in control_names:
    df[name] = state_history[name]

df.to_csv(OUTPUT_FILE, index=False)
print(
    "Master timeline generated successfully. Total simulated time:",
    round(current_time, 2),
    "seconds.",
)
