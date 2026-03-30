# Identifying Deployment Environments and Sensor Nodes Using Link Quality Fluctuations

*Team Members: Duy Do Le (5337109), Ngoc Thao Trang Ho (5316723), Timon Althaus (5280076), Yu-Ching Lai (5328824), Yu Ling Zhong (532226)*

The ability to uniquely identify IoT sensor nodes and their deployment environments is a challenging yet valuable problem in wireless sensing. Due to manufacturing imperfections, individual devices exhibit microscopic differences in their radio characteristics, which can be observed through link quality fluctuations such as RSSI (Received Signal Strength Indicator).

This project investigates whether these fluctuations can be used to classify (1) the deployment environment and (2) the individual sensor node. We deploy a BLE-based sensor network across five outdoor environments and apply two machine learning models (CNN and ResNet) to perform classification.

> [!NOTE]
> The full written project report is available as [**`report.pdf`**](report.pdf) in the repository root.

## Project Structure

- **[iot/](iot/)**: RIOT OS firmware for TX/RX nodes and data collection scripts.
- **[ml/](ml/)**: Data preprocessing, deep learning models (CNN, ResNet), and experimental analysis.
  > **Note:** The required data directories for the project submission (`environment/` and `node/`) correspond to `ml/data/raw-env` and `ml/data/raw-node`.
- **[report/](report/)**: Project documentation and LaTeX report materials.

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
