# EEG Seizure Detection using Topological Data Analysis

## 📌 Overview
This repository implements a machine learning pipeline for the automated detection of epileptic seizures from scalp EEG recordings. The core innovation of this project is the integration of **Topological Data Analysis (TDA)**—specifically Persistent Homology (H1 cycles) - alongside traditional spectral features to capture the shape and connectivity of neural signals.

The system includes a full training pipeline tested on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/#files) and a user-friendly [web interface](https://eeg-analyzer.streamlit.app/) for real-time analysis of EDF files.

## 🚀 Key Features
* **Advanced Preprocessing:** Standardization to the 10-20 channel system.
    * **Independent Component Analysis (ICA)** for artifact removal (EOG/EMG noise).
* **Hybrid Feature Extraction:**
    * **Spectral:** Power Spectral Density (PSD) in Delta, Theta, Alpha, Beta and Gamma bands.
    * **Topological:** H1 Persistent Homology using **Gudhi** (Rips Complex) to map signal loops to persistence landscapes.
    * **Time-Domain:** Line length and signal variance.
* **Robust Classification:** * Random Forest Classifier with SMOTE/RUS for class imbalance handling.
    * Patient-specific Leave-One-Group-Out (LOGO) cross-validation.
    * Post-processing with median filtering to reduce false positives.
* **Interactive Web App:** A Streamlit-based dashboard to visualize raw signals, spectrograms and detection probabilities.

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* Dependencies listed in `web/requirements.txt`

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OleksandrZadvornyi/eeg.git
    cd eeg
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r web/requirements.txt
    ```
    *(Note: Ensure you have `mne`, `gudhi`, `scikit-learn`, `imblearn` and `streamlit` installed).*

## 📊 Data Setup (For Training)
To train the model from scratch, you need the **CHB-MIT Scalp EEG Database**.

1.  Download the dataset from PhysioNet.
2.  Place the patient folders (`chb01`, `chb02`, etc.) inside a directory named `CHB-MIT_Scalp_EEG_Database` in the project root.
3.  Ensure each patient folder contains the `.edf` files and the corresponding `-summary.txt` file.

## 💻 Usage

### 1. Training the Model
Run the main pipeline to process EDF files, extract features and train the Random Forest model. This script performs Leave-One-Group-Out cross-validation and saves the final model.

```bash
python app.py
```

* **Output:** Generates `seizure_model_final.pkl` and `model_config.json`.
* **Note:** The script caches processed features in `chb_features_ica.npz` to speed up subsequent runs.

### 2. Running the Web Dashboard

Launch the interactive tool to analyze specific EDF files without writing code.

```bash
streamlit run web/web_app.py
```

* **Features:**
* Upload any standard `.edf` file.
* Adjust sensitivity thresholds and smoothing kernels dynamically.
* Inspect raw EEG traces and Spectrograms synchronized with detection events.



## 📂 Project Structure

```text
├── app.py                   # Main training script (Feature extraction + CV)
├── seizure_model_final.pkl  # Trained Random Forest model
├── model_config.json        # Saved hyperparameters (thresholds, etc.)
├── utils/                   # Helper scripts for parsing CHB-MIT summaries
|   └── process_edf_file.py  # Core processing logic (ICA, TDA, Spectral)
└── web/                     # Web application files
    ├── web_app.py           # Streamlit entry point
    └── utils/               # Reusable processing modules for the web app

```

## 🔬 Methodology

The pipeline transforms raw EEG data into a high-dimensional feature space:

1. **Signal Cleaning:** High-pass filtering (1Hz) and ICA cleaning.
2. **TDA (Homology):** We treat the PSD as a point cloud and compute the **persistence diagram** of H1 cycles. These diagrams are vectorized into **Persistence Landscapes** to be used as input for the classifier.
3. **Classification:** A Random Forest model detects seizure epochs (5-second windows).
4. **Temporal Smoothing:** A median filter aggregates individual epoch predictions to ensure physiological consistency.

## 📜 License

MIT License

---

*Created as part of a Master's Thesis in Software Engineering.*
