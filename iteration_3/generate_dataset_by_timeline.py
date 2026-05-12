import json
from pathlib import Path

import pandas as pd
from fmpy import extract, read_model_description  # type: ignore
from fmpy.fmi2 import FMU2Slave  # type: ignore
from tqdm import tqdm

# Path Configurations
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

BLOODY_MARY = MODELS_DIR / "modelECMORespiratoryVR_BloodGasesTransport_BloodyMaryPPG2.fmu"
VENTILATOR = MODELS_DIR / "modelECMORespiratoryVR_BloodGasesTransport_LungVentilatorSCMV2.fmu"
HEMODYNAMICS = (
    MODELS_DIR
    / "modelECMORespiratoryVR_BloodGasesTransport_MeursModel2011_HemodynamicsRegulatedHR.fmu"
)

SIMULATION_CONFIG_FILE = BASE_DIR / "simulation_config.json"
SIMULATION_VARIABLES_FILE = BASE_DIR / "simulation_variables.json"
MASTER_TIMELINE_FILE = BASE_DIR / "master_timeline.csv"

# Load Configurations and Timeline
print("Loading master timeline...")
timeline_df = pd.read_csv(MASTER_TIMELINE_FILE)
control_columns = [col for col in timeline_df.columns if col != "time"]

with open(SIMULATION_CONFIG_FILE, "r") as f:
    simulation_config = json.load(f)

with open(SIMULATION_VARIABLES_FILE, "r") as f:
    simulation_variables = json.load(f)

default_state = simulation_config["default_state"]
variables_config = simulation_variables.get("target_vars", {})
controls_config = simulation_variables.get("controls", {})
variables_config.update(controls_config)

ventilator_vars = [var for var, cfg in variables_config.items() if cfg["model"] == "ventilator"]
hemodynamics_vars = [var for var, cfg in variables_config.items() if cfg["model"] == "hemodynamics"]
bloody_mary_vars = [var for var, cfg in variables_config.items() if cfg["model"] == "bloody_mary"]
all_output_vars = list(variables_config.keys())

# Extract & Read FMU Descriptions
UNZIPPED_BLOODY_MARY = extract(str(BLOODY_MARY))
UNZIPPED_VENTILATOR = extract(str(VENTILATOR))
UNZIPPED_HEMODYNAMICS = extract(str(HEMODYNAMICS))

bloody_mary_desc = read_model_description(str(BLOODY_MARY))
ventilator_desc = read_model_description(str(VENTILATOR))
hemodynamics_desc = read_model_description(str(HEMODYNAMICS))

bm_vrs = {var.name: var.valueReference for var in bloody_mary_desc.modelVariables}
vent_vrs = {var.name: var.valueReference for var in ventilator_desc.modelVariables}
hemo_vrs = {var.name: var.valueReference for var in hemodynamics_desc.modelVariables}

# Instantiate FMUs
bloody_mary = FMU2Slave(
    guid=bloody_mary_desc.guid,
    unzipDirectory=UNZIPPED_BLOODY_MARY,
    modelIdentifier=bloody_mary_desc.coSimulation.modelIdentifier,  # type: ignore
    instanceName="bm",
)
ventilator = FMU2Slave(
    guid=ventilator_desc.guid,
    unzipDirectory=UNZIPPED_VENTILATOR,
    modelIdentifier=ventilator_desc.coSimulation.modelIdentifier,  # type: ignore
    instanceName="vent",
)
hemodynamics = FMU2Slave(
    guid=hemodynamics_desc.guid,
    unzipDirectory=UNZIPPED_HEMODYNAMICS,
    modelIdentifier=hemodynamics_desc.coSimulation.modelIdentifier,  # type: ignore
    instanceName="hemo",
)

fmus = [bloody_mary, ventilator, hemodynamics]
model_map = {
    "bloody_mary": (bloody_mary, bm_vrs),
    "ventilator": (ventilator, vent_vrs),
    "hemodynamics": (hemodynamics, hemo_vrs),
}

START_TIME = 0.0
STEP_SIZE = round(1.0 / 30.0, 2)

# Initialize FMUs & Apply Default State
for fmu in fmus:
    fmu.instantiate()
    fmu.setupExperiment(startTime=START_TIME)
    fmu.enterInitializationMode()

initial_controls = timeline_df.iloc[0]

for param_name, config in default_state.items():
    if param_name in control_columns:
        user_value = initial_controls[param_name]
    else:
        user_value = config["value"]

    final_value = (user_value * config["multiplier"]) / config["divider"]
    models = [m.strip() for m in config["model"].split(",")]
    for model_name in models:
        if model_name in model_map and param_name in model_map[model_name][1]:
            fmu_instance, value_references = model_map[model_name]
            fmu_instance.setReal([value_references[param_name]], [final_value])

for fmu in fmus:
    fmu.exitInitializationMode()


# Helper Function
def read_and_convert_variables(
    fmu_instance, value_references, var_names, variables_config, results_dict
):
    for var_name in var_names:
        if var_name in value_references:
            raw_value = fmu_instance.getReal([value_references[var_name]])[0]
            cfg = variables_config[var_name]
            results_dict[var_name].append((raw_value * cfg["multiplier"]) / cfg["divider"])
        else:
            results_dict[var_name].append(None)


# Simulation Loop
results = {var_name: [] for var_name in all_output_vars}
results["time"] = []

so2_vr_bm = bm_vrs["arterial.sO2"]
so2_vr_hemo = hemo_vrs["sO2.k"]

total_steps = len(timeline_df)
progress_bar = tqdm(total=total_steps, desc="Simulating overall timeline", unit="step")

last_applied_controls = {}

for step_idx in range(total_steps):
    current_row = timeline_df.iloc[step_idx]
    current_time = current_row["time"]

    # 1. Update Dynamic Controls
    for param_name in control_columns:
        user_value = current_row[param_name]

        # Delta update to avoid solver spam
        if (
            param_name not in last_applied_controls
            or abs(last_applied_controls[param_name] - user_value) > 1e-6
        ):
            ds = default_state[param_name]
            final_value = (user_value * ds["multiplier"]) / ds["divider"]

            models = [m.strip() for m in ds["model"].split(",")]
            for model_name in models:
                fmu_instance, value_references = model_map[model_name]
                fmu_instance.setReal([value_references[param_name]], [final_value])

            last_applied_controls[param_name] = user_value

    # 2. Step Models
    step_start = round(current_time - STEP_SIZE, 2)

    bloody_mary.doStep(currentCommunicationPoint=step_start, communicationStepSize=STEP_SIZE)
    ventilator.doStep(currentCommunicationPoint=step_start, communicationStepSize=STEP_SIZE)

    # Manual coupling
    arterial_so2 = bloody_mary.getReal([so2_vr_bm])[0]
    hemodynamics.setReal([so2_vr_hemo], [arterial_so2])

    hemodynamics.doStep(currentCommunicationPoint=step_start, communicationStepSize=STEP_SIZE)

    # 3. Record Results
    results["time"].append(round(current_time, 2))
    read_and_convert_variables(ventilator, vent_vrs, ventilator_vars, variables_config, results)
    read_and_convert_variables(hemodynamics, hemo_vrs, hemodynamics_vars, variables_config, results)
    read_and_convert_variables(bloody_mary, bm_vrs, bloody_mary_vars, variables_config, results)

    progress_bar.update(1)

progress_bar.close()

# Terminate and Save
for fmu in fmus:
    fmu.terminate()
    fmu.freeInstance()

df_results = pd.DataFrame(results)
other_cols = sorted([col for col in df_results.columns if col != "time"])
df_results = df_results[["time"] + other_cols]

output_file = DATA_DIR / "simulation_dataset_dynamic.csv"
df_results.to_csv(output_file, index=False)
print(f"Simulation complete. Results saved to {output_file.name}")
