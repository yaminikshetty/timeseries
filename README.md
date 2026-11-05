# Nuclear Baseload Anomaly Detection & Time-Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: PUR-1](https://img.shields.io/badge/Dataset-PUR--1%20Reactor-success.svg)](https://github.com/zachdahm/Nuclear-data-TS-forecasting.git)

Replicating, benchmarking, and advancing state-of-the-art anomaly detection for nuclear reactor digital Instrumentation and Control (I&C) systems using high-frequency time-series telemetry from the **Purdue University Reactor Number One (PUR-1)** — the first fully digital university research reactor licensed in the United States.

This project replicates the baseline GRU model from **Zachery Dahm's Master's Thesis** (*"Development of a Time-Series Forecasting Model for Detecting Anomalies in Nuclear Reactor Data"*, Purdue University, May 2024) and extends it with advanced architectures, rigorous anomaly detection benchmarks, multi-class diagnostic classification, and deep explainability.

---

## 🔬 Core Physics & Telemetry

PUR-1 telemetry records continuous operational data at **1 Hz (1-second sampling rate)** across **8 primary sensor channels**:

| Feature Name | Description | Physical Role |
| :--- | :--- | :--- |
| `nfd-1-cps` | Neutron Flux Detector 1 (Counts per Second) | Core power level / prompt neutron population |
| `nfd-1-cr` | Neutron Flux Detector 1 (Count Rate) | Rate of power change / reactor period proxy |
| `rr-active-state` | Regulating Rod Active State | Discrete status indicator (0: stationary, 1: moving) |
| `rr-position` | Regulating Rod Position | Fine reactivity control mechanism (normalized position) |
| `ss1-active-state`| Safety Shim Rod 1 Active State | Discrete status indicator (0: stationary, 1: moving) |
| `ss1-position` | Safety Shim Rod 1 Position | Coarse reactivity control and shutdown capability |
| `ss2-active-state`| Safety Shim Rod 2 Active State | Discrete status indicator (0: stationary, 1: moving) |
| `ss2-position` | Safety Shim Rod 2 Position | Redundant coarse reactivity control mechanism |

---

## 🎯 5 Operational Regimes & Anomaly Classes

Unlike standard binary anomaly detection, this framework differentiates between benign operational variations, safety shutdowns, and malicious attacks:

1. **Normal Baseload Operation (`real_dataset_normalized.csv`, 200,000 steps)**: Steady-state nuclear operation. The model is trained *strictly* on normal data under the one-class classification paradigm.
2. **Operational Transients (`transient_normal.csv`, 19,784 steps)**: Benign, planned power maneuvers and control rod adjustments that must **not** trigger false alarms.
3. **Emergency Scrams (`scrams_normal.csv`, 19,784 steps)**: Rapid gravity insertion of safety shim rods due to emergency shutdown trips. Characterized by rapid simultaneous rod collapse and subcritical neutron flux decay.
4. **False Data Injection Type 1 (`FDI1_normal.csv`, 19,784 steps)**: Cyberattack featuring continuous, subtle linear drift injected onto neutron flux telemetry (`nfd-1-cps` and `nfd-1-cr`) while rod positions remain unperturbed.
5. **False Data Injection Type 2 (`FDI2_normal.csv`, 19,784 steps)**: Cyberattack featuring abrupt step alterations and scaling injected onto flux telemetry while rod states reflect steady-state baseload.

---

## 🚀 Key Improvements Over Dahm (2024)

| Limitation in Dahm (2024) | Improvement in This Framework |
| :--- | :--- |
| **Naive Sliding Windows**: Sliced across reactor restart boundaries, creating artificial prediction spikes. | **Run Boundary Awareness**: Validates $\Delta \text{index} == 1$ across every 35s window, skipping 39 daily shutdown/restart boundaries. |
| **Single Metric Evaluation**: Only reported raw RMSE on test sets. | **Industrial Benchmark Metrics**: AUROC, AUPRC, F1-Score, Precision, Recall, PR curves, and statistical thresholds ($3\sigma$, 95th, 99th percentiles). |
| **Model Scope**: Limited to single-layer GRU vs basic LSTM/ANN. | **Advanced Deep Architectures**: Stacked BiLSTM, Temporal Convolutional Networks (TCN), Time-Series Transformer, and LSTM-Autoencoder. |
| **Binary Thresholding**: Could not diagnose anomaly root causes. | **Two-Stage Diagnostic Classifier**: LightGBM trained on 18 physical/residual features to distinguish between Scram, Transient, FDI-1, and FDI-2. |
| **Black Box**: No feature attribution or operator explanation. | **Operator Explainability**: Deep SHAP attribution, self-attention temporal heatmaps, Captum Integrated Gradients, and automated incident advisory reports. |

---

## 📂 Notebook Pipeline

The project is structured into 6 sequential, fully runnable Jupyter notebooks:

### [`00_data_exploration.ipynb`](00_data_exploration.ipynb) — Exploratory Data Analysis
- Loads all 5 operational datasets (279,136 total seconds).
- Validates index continuity and visualizes the **40 continuous reactor runs** via operational timeline sawtooth plots.
- Cross-dataset statistical comparison and $\Delta$ mean heatmap vs normal operation.
- Multi-signal 4×2 grid subplots and interactive Plotly anomaly fingerprint overlay.

### [`01_baseline_gru.ipynb`](01_baseline_gru.ipynb) — Baseline GRU Replication
- Replicates Dahm’s exact PyTorch GRU architecture (100 hidden units, lookback $N=30$, lookahead $F=5$, Adam optimizer).
- Implements Run-Boundary Aware temporal 60 / 20 / 20 data splitting.
- Trains with early stopping and validates against Dahm’s reported RMSE.
- Benchmarks baseline thresholding on Scrams and Cyberattacks.

### [`02_improved_models.ipynb`](02_improved_models.ipynb) — Advanced Architectures
- **Stacked BiLSTM**: Bidirectional recurrent processing with inter-layer dropout.
- **Temporal Convolutional Network (TCN)**: Dilated causal convolutions with residual connections ($2^i$ receptive field expansion).
- **Time-Series Transformer**: Multi-head self-attention ($d_{\text{model}}=64$, 4 heads, positional encodings).
- **LSTM-Autoencoder**: Unsupervised multivariate reconstruction error across all 8 sensor channels.

### [`03_evaluation_metrics.ipynb`](03_evaluation_metrics.ipynb) — Benchmark Evaluation
- Calculates **AUROC**, **AUPRC**, **F1-Score**, Precision, and Recall across all evaluation datasets.
- ROC Curves and Precision-Recall Curves overlaying all 5 architectures.
- Threshold sensitivity analysis: $3\sigma$, 95th percentile, and 99th percentile normal operating limits.
- Mann-Whitney U statistical significance testing.

### [`04_anomaly_classification.ipynb`](04_anomaly_classification.ipynb) — Multi-Class Diagnostic Engine
- Two-stage diagnostic engine: Stage 1 detects anomaly presence via forecaster residual; Stage 2 classifies root cause.
- Feature engineering: 18 physical-residual features (flux error, rod motion coherence, residual variance, flux-rod correlation).
- Trains a LightGBM multi-class model with 5-fold stratified cross-validation.
- Generates 5×5 confusion matrix, classification reports, and physical rule boundary analysis.

### [`05_explainability.ipynb`](05_explainability.ipynb) — Explainability & Incident Advisory
- **SHAP (SHapley Additive exPlanations)**: KernelExplainer feature attributions across timesteps and sensors.
- **Transformer Self-Attention Heatmaps**: Reveals temporal focus across the 30-second lookback window.
- **Captum Integrated Gradients**: Axiomatic attribution for deep recurrent layers.
- **Automated Operator Incident Report**: Generates structured, timestamped markdown incident advisories with root-cause diagnoses, affected channels, and actionable mitigation guidance.

---

## ⚡ Quickstart

### Running in Google Colab (Recommended for GPU)
1. Open any notebook in Google Colab.
2. Select **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
3. Run the setup cell at the top of the notebook; the dataset will be cloned automatically from the repository.

### Local Installation
```bash
# Clone the repository
git clone https://github.com/yaminikshetty/timeseries.git
cd timeseries

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchinfo scikit-learn pandas numpy matplotlib seaborn plotly lightgbm shap captum scipy tqdm
```

---

## 📊 Dataset Structure
The `Nuclear-data-TS-forecasting/` directory contains normalized CSV telemetry from PUR-1:
- `real_dataset_normalized.csv`: 200,000 steps of normal reactor operation.
- `transient_normal.csv`: 19,784 steps of planned power transients.
- `scrams_normal.csv`: 19,784 steps of emergency reactor scrams.
- `FDI1_normal.csv`: 19,784 steps of False Data Injection Cyberattack Type 1 (drift).
- `FDI2_normal.csv`: 19,784 steps of False Data Injection Cyberattack Type 2 (step).
- `GRU_Final.pt`: Dahm's pre-trained PyTorch GRU checkpoint for direct verification.

---

## 📜 References
- **Dahm, Z.** (2024). *Development of a Time-Series Forecasting Model for Detecting Anomalies in Nuclear Reactor Data*. Master's Thesis, School of Nuclear Engineering, Purdue University.
- **Purdue University Reactor Number One (PUR-1)**: All-digital instrumentation and control (I&C) research reactor operational datasets.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
