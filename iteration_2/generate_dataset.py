import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from tqdm import tqdm

# ---------------------------------------------
# Path Configurations
# ---------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

BLOODY_MARY = MODELS_DIR / "modelECMORespiratoryVR_BloodGasesTransport_BloodyMaryPPG2.fmu"
VENTILATOR = MODELS_DIR / "modelECMORespiratoryVR_BloodGasesTransport_LungVentilatorSCMV2.fmu"
HEMODYNAMICS = (
    MODELS_DIR
    / "modelECMORespiratoryVR_BloodGasesTransport_MeursModel2011_HemodynamicsRegulatedHR.fmu"
)

SIMULATION_VARIABLES_FILE = BASE_DIR / "simulation_variables.json"

# ---------------------------------------------
# Simulation Settings
# ---------------------------------------------

PATIENT_STATE = "pneumonia"  # Options: "normal", "pneumonia", "ventilated", "stabilized"
SIMULATE_CONTROLS = True  # Whether to include control variables in the simulation
PLOT_RESULTS = False

START_TIME = 0.0
STOP_TIME = 3600
SAMPLING_RATE = 30

STEP_SIZE = round((1.0 / SAMPLING_RATE), 2)

# ---------------------------------------------
# Load Simulation Variable Configurations
# ---------------------------------------------

# Load monitor variables from JSON
with open(SIMULATION_VARIABLES_FILE, "r") as f:
    simulation_variables = json.load(f)

# Parse simulation variables
variables_config = simulation_variables.get("monitor", {})
if SIMULATE_CONTROLS:
    controls_config = simulation_variables.get("controls", {})
    variables_config.update(controls_config)

plot_vars = simulation_variables.get("plot_variables", [])

# Group variables by model for efficient reading
ventilator_vars = [
    var for var, config in variables_config.items() if config["model"] == "ventilator"
]
hemodynamics_vars = [
    var for var, config in variables_config.items() if config["model"] == "hemodynamics"
]
bloody_mary_vars = [
    var for var, config in variables_config.items() if config["model"] == "bloody_mary"
]

# Collect all monitor variable names
all_monitor_vars = list(variables_config.keys())

# Load patient states from JSON
PATIENT_STATES_FILE = BASE_DIR / "patient_states.json"
with open(PATIENT_STATES_FILE, "r") as f:
    patient_states = json.load(f)

# Get the configuration for the selected patient state
if PATIENT_STATE not in patient_states.get("states", {}):
    raise ValueError(f"Patient state '{PATIENT_STATE}' not found in {PATIENT_STATES_FILE}")

UNZIPPED_BLOODY_MARY = extract(str(BLOODY_MARY))
UNZIPPED_VENTILATOR = extract(str(VENTILATOR))
UNZIPPED_HEMODYNAMICS = extract(str(HEMODYNAMICS))

# Read model descriptions
bloody_mary_description = read_model_description(str(BLOODY_MARY))
lung_ventilator_description = read_model_description(str(VENTILATOR))
hemodynamics_description = read_model_description(str(HEMODYNAMICS))

# Get value references for all model variables
bloody_mary_value_references = {
    var.name: var.valueReference for var in bloody_mary_description.modelVariables
}
lung_ventilator_value_references = {
    var.name: var.valueReference for var in lung_ventilator_description.modelVariables
}
hemodynamics_value_references = {
    var.name: var.valueReference for var in hemodynamics_description.modelVariables
}

# ---------------------------------------------
# Initialize FMUs & Apply Patient State
# ---------------------------------------------

# Instantiate the FMUs for Co-Simulation
bloody_mary = FMU2Slave(
    guid=bloody_mary_description.guid,
    unzipDirectory=UNZIPPED_BLOODY_MARY,
    modelIdentifier=bloody_mary_description.coSimulation.modelIdentifier,  # type: ignore
    instanceName="instance",
)

ventilator = FMU2Slave(
    guid=lung_ventilator_description.guid,
    unzipDirectory=UNZIPPED_VENTILATOR,
    modelIdentifier=lung_ventilator_description.coSimulation.modelIdentifier,  # type: ignore
    instanceName="instance",
)

hemodynamics = FMU2Slave(
    guid=hemodynamics_description.guid,
    unzipDirectory=UNZIPPED_HEMODYNAMICS,
    modelIdentifier=hemodynamics_description.coSimulation.modelIdentifier,  # type: ignore
    instanceName="instance",
)

fmus = [bloody_mary, ventilator, hemodynamics]

for fmu in fmus:
    fmu.instantiate()
    fmu.setupExperiment(startTime=START_TIME)
    fmu.enterInitializationMode()

# Apply patient state configuration
print(f"\nApplying patient state: {PATIENT_STATE}")
parameters_config = patient_states["parameters"]
state_values = patient_states["states"][PATIENT_STATE]

# Map model names to FMU instances and their value references
model_map = {
    "bloody_mary": (bloody_mary, bloody_mary_value_references),
    "ventilator": (ventilator, lung_ventilator_value_references),
    "hemodynamics": (hemodynamics, hemodynamics_value_references),
}

for param_label, param_value in state_values.items():
    if param_label not in parameters_config:
        print(f"Warning: Parameter '{param_label}' not found in parameters configuration")
        continue

    param_info = parameters_config[param_label]
    param_name = param_info["key"]
    multiplier = param_info["multiplier"]
    divider = param_info["divider"]
    models = [m.strip() for m in param_info["model"].split(",")]

    # Apply multiplication first, then division
    final_value = (param_value * multiplier) / divider

    print(f"Setting {param_name} = {final_value} (from {param_value} * {multiplier} / {divider})")

    for model_name in models:
        if model_name in model_map:
            fmu_instance, value_references = model_map[model_name]
            if param_name in value_references:
                vr = value_references[param_name]
                fmu_instance.setReal([vr], [final_value])
            else:
                print(f"Warning: '{param_name}' not found in {model_name} FMU")
        else:
            print(f"Warning: Unknown model '{model_name}'")

for fmu in fmus:
    fmu.exitInitializationMode()

# Initialize storage for results (monitor variables + patient state)
results = {var_name: [] for var_name in all_monitor_vars}
results["time"] = []
results["patient_state_id"] = []

# Get the current patient state ID
current_patient_state_id = patient_states["patient_state_ids"][PATIENT_STATE]

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------


def read_and_convert_variables(
    fmu_instance, value_references, var_names, variables_config, results
):
    """Read variables from FMU, apply conversions, and store in results."""
    for var_name in var_names:
        if var_name in value_references:
            vr = value_references[var_name]
            raw_value = fmu_instance.getReal([vr])[0]

            # Apply conversion
            var_config = variables_config[var_name]
            converted_value = (raw_value * var_config["multiplier"]) / var_config["divider"]
            results[var_name].append(converted_value)
        else:
            results[var_name].append(None)


# ---------------------------------------------
# Simulation Loop
# ---------------------------------------------

# Initialize progress bar
total_steps = int((STOP_TIME - START_TIME) / STEP_SIZE) + 1
progress_bar = tqdm(total=total_steps, desc="Simulating", unit="step")

current_time = START_TIME
while current_time < STOP_TIME:
    # Advance simulation for all FMUs
    bloody_mary.doStep(currentCommunicationPoint=current_time, communicationStepSize=STEP_SIZE)
    ventilator.doStep(currentCommunicationPoint=current_time, communicationStepSize=STEP_SIZE)
    hemodynamics.doStep(currentCommunicationPoint=current_time, communicationStepSize=STEP_SIZE)

    # Store current time and patient state
    results["time"].append(round(current_time, 2))
    results["patient_state_id"].append(current_patient_state_id)

    # Read and convert variables from all FMUs
    read_and_convert_variables(
        ventilator, lung_ventilator_value_references, ventilator_vars, variables_config, results
    )
    read_and_convert_variables(
        hemodynamics, hemodynamics_value_references, hemodynamics_vars, variables_config, results
    )
    read_and_convert_variables(
        bloody_mary, bloody_mary_value_references, bloody_mary_vars, variables_config, results
    )

    current_time += STEP_SIZE
    progress_bar.update(1)

progress_bar.close()

# Terminate FMUs
for fmu in fmus:
    fmu.terminate()
    fmu.freeInstance()

# ---------------------------------------------
# Save Results & Plotting
# ---------------------------------------------

# Save results to CSV
df = pd.DataFrame(results)
output_file = (
    DATA_DIR
    / f"simulation_{PATIENT_STATE}_{'with_controls' if SIMULATE_CONTROLS else 'without_controls'}_{STOP_TIME}s.csv"
)

# Reorder columns to have 'time' and 'patient_state_id' first, then sort the rest alphabetically
other_cols = sorted([col for col in df.columns if col not in ["time", "patient_state_id"]])
cols = ["time", "patient_state_id"] + other_cols
df = df[cols]

df.to_csv(output_file, index=False)
print(f"\nSimulation complete. Results saved to: {DATA_DIR}/{output_file.name}")

# Plot the specified variables
if PLOT_RESULTS:
    fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 2.5 * len(plot_vars)))
    fig.suptitle(f"Simulation Results for {PATIENT_STATE.capitalize()}", fontsize=16)

    for idx, var_name in enumerate(plot_vars):
        ax = axes[idx] if len(plot_vars) > 1 else axes
        if var_name in df.columns:
            ax.plot(df["time"], df[var_name], linewidth=1)
            ax.set_ylabel(var_name)
            ax.grid(True, alpha=0.3)
            if idx == len(plot_vars) - 1:
                ax.set_xlabel("Time (s)")
        else:
            ax.text(
                0.5,
                0.5,
                f"Variable '{var_name}' not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel(var_name)

    plt.tight_layout()
    plot_file = PLOTS_DIR / f"simulation_{PATIENT_STATE}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plots saved to: {plot_file}")
    # plt.show()
