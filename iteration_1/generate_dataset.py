from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from tqdm import tqdm

# Path to FMU
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FMU_PATH = MODELS_DIR / "hemodynamic_model_burkhoff.fmu"
UNZIPPED_FMU = extract(str(FMU_PATH))

# Simulation settings
START_TIME = 0.0
STOP_TIME = 300.0
SAMPLING_RATE = 30  # Hz
STEP_SIZE = round((1.0 / SAMPLING_RATE), 2)

# Read model description
model_description = read_model_description(str(FMU_PATH))

# Get value references for all model variables
value_references = {var.name: var.valueReference for var in model_description.modelVariables}

# Instantiate the FMU for Co-Simulation
fmu = FMU2Slave(
    guid=model_description.guid,
    unzipDirectory=UNZIPPED_FMU,
    modelIdentifier=model_description.coSimulation.modelIdentifier,  # type: ignore
    instanceName="instance",
)

fmu.instantiate()

# Set initial heart rate
HEART_RATE = 90  # bpm
heart_rate_hz = HEART_RATE / 60

heart_rate_ref = value_references.get("heartRate.k")
if heart_rate_ref is not None:
    fmu.setReal([heart_rate_ref], [heart_rate_hz])
else:
    raise ValueError("Variable 'heartRate.k' not found in the FMU.")

# Setup and initialize
fmu.setupExperiment(startTime=START_TIME, stopTime=STOP_TIME)
fmu.enterInitializationMode()
fmu.exitInitializationMode()

# Initialize storage for results
results = {var_name: [] for var_name in value_references}
results["time"] = []

# Initialize progress bar
total_steps = int((STOP_TIME - START_TIME) / STEP_SIZE) + 1
progress_bar = tqdm(total=total_steps, desc="Simulating", unit="step")

# Simulation loop
current_time = START_TIME
while current_time <= STOP_TIME:
    fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=STEP_SIZE)

    results["time"].append(round(current_time, 2))

    # Read all variable values
    for var_name, var_ref in value_references.items():
        try:
            value = fmu.getReal([var_ref])[0]
        except Exception as e:
            print(f"Error reading variable '{var_name}': {e}")
            break
        results[var_name].append(value)

    current_time += STEP_SIZE
    progress_bar.update(1)  # Update progress bar

# Close progress bar
progress_bar.close()

# Terminate and free resources
fmu.terminate()
fmu.freeInstance()

# Save to CSV file
output_csv_path = (
    BASE_DIR / f"simulated_data/{SAMPLING_RATE}hz/default_{HEART_RATE}_bpm_{SAMPLING_RATE}hz.csv"
)
dataset = pd.DataFrame(results)

# Reorder columns to make 'time' the first column and sort the rest alphabetically
columns = ["time"] + sorted([col for col in dataset.columns if col != "time"])
dataset = dataset[columns]

# Create directory if it doesn't exist
output_csv_path.parent.mkdir(parents=True, exist_ok=True)

dataset.to_csv(output_csv_path, index=False)
print(f"✅ Simulation data saved to '{output_csv_path}'")


# Plotting few variables
plt.figure(figsize=(10, 5))

if "LA.q_in.pressure" in dataset:
    plt.plot(dataset["time"], dataset["LA.q_in.pressure"], label="Left Atrium Pressure")
if "LV.q_in.pressure" in dataset:
    plt.plot(dataset["time"], dataset["LV.q_in.pressure"], label="Left Ventricle Pressure")
if "AOV.q_out.pressure" in dataset:
    plt.plot(dataset["time"], dataset["AOV.q_out.pressure"], label="Aortic Valve Pressure")

# Show heart rate in legend
plt.plot([], [], " ", label=f"Heart rate: {HEART_RATE} bpm")

plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Simulation Output")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
