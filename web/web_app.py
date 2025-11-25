import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import tempfile
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from utils.process_edf_file import process_edf_file
# Import your existing processing logic
# Assumes process_edf_file.py is inside a folder named 'utils'
# --- Page Configuration ---
st.set_page_config(
    page_title="EEG Seizure Detector",
    page_icon="🧠",
    layout="wide"
)
import sys
st.write(sys.executable)

# try:
    
# except ImportError:
#     st.error("Could not import 'process_edf_file'. Ensure 'utils/process_edf_file.py' exists.")



st.title("🧠 EEG Seizure Detection System")
st.markdown("""
Upload an **EDF file** to detect seizure activity using the trained Machine Learning model.
The system analyzes Spectral, TDA (Topological), and Signal features.
""")

# --- Sidebar: Model Status ---
st.sidebar.header("Model Status")

@st.cache_resource
def load_model_and_config():
    """
    Loads the model and config once and caches them in memory 
    so we don't reload them on every interaction.
    """
    try:
        model = joblib.load('seizure_model_final.pkl')
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        return model, config
    except FileNotFoundError:
        return None, None

model, config = load_model_and_config()

if model is not None:
    st.sidebar.success("✅ Model Loaded")
    st.sidebar.markdown(f"**Threshold:** {config['best_threshold']:.4f}")
    st.sidebar.markdown(f"**Smoothing Kernel:** {config['filter_kernel']}")
else:
    st.sidebar.error("❌ Model not found. Please ensure .pkl and .json files are in the directory.")
    st.stop()

# --- Main Logic ---

uploaded_file = st.file_uploader("Choose an EDF file...", type=['edf'])

if uploaded_file is not None:
    # 1. Save uploaded file to a temporary path (MNE requires a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info(f"Processing **{uploaded_file.name}**... This may take a while depending on TDA complexity.")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    try:
        # 2. Process the file using YOUR existing function
        # We pass an empty list [] for seizure_intervals because we are in INFERENCE mode.
        # We don't know where the seizures are yet!
        features, _ = process_edf_file(tmp_file_path, seizure_intervals=[])
        
        progress_bar.progress(50)
        
        if features is None:
            st.error("Processing failed. The file might not have the required channels.")
        else:
            # 3. Make Predictions
            st.write("Running Inference...")
            
            # Get raw probabilities (Confidence scores)
            probs = model.predict_proba(features)[:, 1]
            
            # Apply the optimized Threshold
            threshold = config['best_threshold']
            initial_preds = (probs >= threshold).astype(int)
            
            # Apply the Median Filter (Smoothing)
            kernel = config['filter_kernel']
            final_preds = medfilt(initial_preds, kernel_size=kernel)
            
            progress_bar.progress(100)
            
            # --- Results Display ---
            
            # A. Summary Metrics
            n_seizure_epochs = np.sum(final_preds)
            total_epochs = len(final_preds)
            duration_sec = total_epochs * 5  # 5 seconds per epoch
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Duration", f"{duration_sec/60:.1f} min")
            col2.metric("Seizure Burden", f"{n_seizure_epochs * 5} sec")
            
            status = "⚠️ Seizure Detected" if n_seizure_epochs > 0 else "✅ Normal Record"
            col3.markdown(f"### {status}")
            
            # B. Visualization (Probability Plot)
            st.subheader("Seizure Probability over Time")
            
            # Create a time axis (in minutes)
            time_axis = np.arange(total_epochs) * 5 / 60 
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Plot raw probability
            ax.plot(time_axis, probs, color='gray', alpha=0.3, label='Raw Probability')
            
            # Highlight detected regions
            ax.fill_between(time_axis, 0, 1, where=(final_preds == 1), color='red', alpha=0.3, label='Detected Seizure')
            
            # Add Threshold line
            ax.axhline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.2f})')
            
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Seizure Probability")
            ax.set_ylim(0, 1.05)
            ax.legend(loc='upper right')
            
            st.pyplot(fig)
            
            # C. Detailed Timestamp Table
            if n_seizure_epochs > 0:
                st.subheader("Detailed Detections")
                
                # Convert binary array to start-end intervals
                # Logic: Find where signal flips from 0 to 1 and 1 to 0
                padded = np.pad(final_preds, (1, 1), 'constant')
                diffs = np.diff(padded)
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                
                detections = []
                for s, e in zip(starts, ends):
                    start_time = s * 5
                    end_time = e * 5
                    detections.append({
                        "Start Time (s)": start_time,
                        "End Time (s)": end_time,
                        "Duration (s)": end_time - start_time,
                        "Start (min)": f"{start_time/60:.2f}",
                        "End (min)": f"{end_time/60:.2f}"
                    })
                
                df_results = pd.DataFrame(detections)
                st.dataframe(df_results)
                
                # Download Button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Detection Report",
                    csv,
                    "seizure_report.csv",
                    "text/csv",
                    key='download-csv'
                )
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Cleanup temp file
        os.remove(tmp_file_path)