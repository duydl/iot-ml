# RSSI-based Localization & Environmental Classification

This project implements a system for collecting and classifying Bluetooth Low Energy (BLE) signal strength (RSSI) data. It consists of an IoT data collection module and a Machine Learning classification module.

## Project Structure

- **[🛰️ iot/](iot/)**: RIOT OS firmware for TX/RX nodes and data collection scripts.
- **[🧠 ml/](ml/)**: Data preprocessing, deep learning models (CNN, ResNet), and experimental analysis.
- **[📊 report/](report/)**: Project documentation and LaTeX report materials.

## Getting Started

### 1. Requirements
Ensure you have the following installed:
- **RIOT OS dependencies** (for IoT module).
- **[uv](https://github.com/astral-sh/uv)** (for ML module).

### 2. Environment Setup
Initialize the Python environment and install all dependencies (ML and IoT tools):
```bash
uv sync --all-extras
```

### 3. Detailed Instructions
Please refer to the subdirectory-specific documentation for detailed setup and usage:
- **[IoT Setup and Data Collection](iot/README.md)**
- **[ML Training and Evaluation](ml/README.md)**

## Workflow Overview
1. **IoT**: Flash nodes and collect RSSI data in various environments.
2. **ML**: Preprocess data into windows and train models to recognize either the transmitter node or the physical environment.
3. **Report**: Analyze results and document findings.
