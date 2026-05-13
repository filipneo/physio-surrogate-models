# Physio Surrogate Models

This repository contains the source code for the master's thesis **"Utilization of Deep Neural Networks for Modeling Medical Devices and Patient Physiological Parameters in Virtual Reality"**.

The goal of the project is to transform computationally expensive mathematical models of physiology (from the eGolem platform) into fast, interactive data-driven surrogate models. These models are optimized for real-time deployment in a virtual reality simulation environment (VR-JIP).

## Repository Structure

The development was iterative, and the repository is divided into three main parts reflecting this process:

* **`iteration_1/`**: Isolated simulation of hemodynamics using a 1D Residual Convolutional Network (RCN).
* **`iteration_2/`**: Deployment of a Transformer architecture for a more complex simulation of a patient with pneumonia (modeling fixed discrete pathological states).
* **`iteration_3/`**: The final and most complex fully continuous Transformer model. It enables smooth real-time control of parameters (e.g., total lung compliance, left-to-right shunt, dead space volume).

Each folder contains:
* Scripts for running mathematical solvers (FMU models) and generating data.
* Neural network architecture definitions and training loops (PyTorch).
* Previews of the generated training datasets in CSV format.

The **`utils/`** folder contains shared helper scripts and functions for data processing and evaluation.

## Execution and Dependencies

The project uses the modern **`uv`** tool for dependency management. The environment configuration is defined in the `pyproject.toml` and `uv.lock` files.

Due to the strict compatibility of the provided FMU models with the Linux x86_64 architecture, the data generation process (Co-Simulation) is containerized using **Docker** (see `Dockerfile` and `docker-compose.yml` in the respective iterations).

## Production Deployment

The final model from the third iteration was exported to the ONNX format and encapsulated into a web component. It is publicly available as an NPM package for easy integration into web and VR interfaces:
* **NPM**: `pneumonia-surrogate`

## License

This project is licensed under the MIT License - see the `LICENSE` file.
