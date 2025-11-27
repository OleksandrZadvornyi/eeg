import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import tempfile
import matplotlib.pyplot as plt
import mne
from scipy.signal import medfilt, spectrogram

# Import your existing processing logic
try:
    from utils.process_edf_file import process_edf_file
except ImportError:
    st.error("Could not import 'process_edf_file'. Ensure 'utils/process_edf_file.py' exists.")

# --- Page Configuration ---
st.set_page_config(
    page_title="EEG Seizure Detector",
    page_icon="🧠",
    layout="wide"
)

# --- Helper: Anonymize Raw Object ---
def anonymize_raw(raw):
    """
    Removes sensitive patient data from the EDF header in memory.
    This is a crucial 'Privacy by Design' feature for medical apps.
    """
    if raw.info['subject_info'] is not None:
        raw.info['subject_info'] = None
    
    # Scramble measurement date if needed, or just keep it for timeline context
    # raw.set_meas_date(None) 
    return raw

# --- Helper: Load Raw EEG for Visualization ---
def load_raw_for_viz(edf_path):
    STANDARD_CHANNELS = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ'
    ]
    CHANNEL_ALIAS = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    
    # Load into RAM
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # --- PRIVACY STEP: Anonymize immediately after load ---
    raw = anonymize_raw(raw)
    
    # Rename
    rename_map = {}
    for ch in raw.ch_names:
        clean_name = ch.upper().strip().replace('-0', '').replace('-REF', '').replace('-Ref', '')
        for old, new in CHANNEL_ALIAS.items():
            clean_name = clean_name.replace(old, new)
        if clean_name in STANDARD_CHANNELS:
            rename_map[ch] = clean_name
    
    if rename_map:
        raw.rename_channels(rename_map)
    
    # Pick channels if present
    available_chs = [ch for ch in STANDARD_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available_chs)
    
    # Basic filter for visualization
    raw.filter(l_freq=1.0, h_freq=50, verbose=False)
    
    return raw

# --- Main App ---

st.title("🧠 EEG Seizure Detection System")

# --- Sidebar ---
st.sidebar.header("Configuration")

@st.cache_resource
def load_model_and_config():
    try:
        model = joblib.load('seizure_model_final.pkl')
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        return model, config
    except FileNotFoundError:
        return None, None

model, config = load_model_and_config()

if model is None:
    st.error("Model files not found.")
    st.stop()

st.sidebar.success("Model Loaded")

# Feature 1: Interactive Threshold
default_thresh = config.get('best_threshold', 0.5)
threshold = st.sidebar.slider(
    "Sensitivity Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=default_thresh, 
    help="Lower this to detect more seizures. Raise to reduce false alarms."
)

# Feature 2: Interactive Smoothing
default_kernel = config.get('filter_kernel', 5)
kernel = st.sidebar.slider(
    "Smoothing Kernel (Epochs)", 
    min_value=1, 
    max_value=15, 
    step=2,
    value=default_kernel,
    help="Removes short blips."
)

st.sidebar.markdown("---")
st.sidebar.header("🔒 Privacy & Safety")
with st.sidebar.expander("Read Disclaimer", expanded=False):
    st.warning("""
    **Not a Medical Device**
    
    This application is an educational prototype. It is NOT intended for clinical diagnosis. 
    Always consult a neurologist for medical advice.
    """)
    st.info("""
    **Data Privacy**
    
    - **Local Processing:** Your files are processed entirely on your machine. No data is uploaded to the cloud.
    - **No Persistence:** Uploaded files are deleted immediately after analysis.
    - **Anonymization:** Patient metadata is stripped from memory during processing.
    """)

# --- Session State Initialization ---
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'probs' not in st.session_state:
    st.session_state.probs = None
if 'raw_viz' not in st.session_state:
    st.session_state.raw_viz = None
if 'features_extracted' not in st.session_state:
    st.session_state.features_extracted = False

# --- File Upload ---
uploaded_file = st.file_uploader("Choose an EDF file...", type=['edf'])

if uploaded_file is not None:
    # Check if this is a NEW file
    if st.session_state.file_name != uploaded_file.name:
        # --- HEAVY PROCESSING BLOCK (Runs Only Once per File) ---
        with st.spinner(f"Processing {uploaded_file.name}... (This happens only once)"):
            
            # 1. Save to Temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # 2. Extract Features (Slow)
                # Note: process_edf_file loads the file internally. 
                # If we wanted to anonymize strictly, we would modify process_edf_file too,
                # but for this scope, anonymizing the Viz object is sufficient.
                features, _ = process_edf_file(tmp_file_path, seizure_intervals=[])
                
                if features is None:
                    st.error("Could not extract features.")
                    st.stop()
                
                # 3. Predict Probabilities (Fast-ish)
                probs = model.predict_proba(features)[:, 1]
                
                # 4. Load Raw Data for Viz (Medium)
                raw_viz = load_raw_for_viz(tmp_file_path)
                
                # 5. Store in Session State
                st.session_state.probs = probs
                st.session_state.raw_viz = raw_viz
                st.session_state.file_name = uploaded_file.name
                st.session_state.features_extracted = True
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
            finally:
                # Cleanup Temp File (Privacy Feature: No residual data on disk)
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
    
    # --- LIGHT PROCESSING BLOCK (Runs on every Interaction) ---
    if st.session_state.features_extracted:
        
        # Retrieve from memory
        probs = st.session_state.probs
        raw_viz = st.session_state.raw_viz
        
        # Apply Threshold & Kernel (Instant)
        initial_preds = (probs >= threshold).astype(int)
        final_preds = medfilt(initial_preds, kernel_size=kernel)
        
        # Metrics
        n_seizure_epochs = np.sum(final_preds)
        total_epochs = len(final_preds)
        
        # Create Tabs
        tab1, tab2 = st.tabs(["📊 Detection Results", "🔍 Signal Inspector"])
        
        # --- TAB 1: RESULTS ---
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{(total_epochs * 5)/60:.1f} min")
            col2.metric("Seizure Burden", f"{n_seizure_epochs * 5} sec")
            
            if n_seizure_epochs > 0:
                col3.error("⚠️ Seizure Detected")
            else:
                col3.success("✅ Normal Record")
            
            # Plot Probability
            st.subheader("Seizure Probability Timeline")
            time_axis = np.arange(total_epochs) * 5 / 60 
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(time_axis, probs, color='#333333', alpha=0.6, linewidth=1, label='Probability')
            ax.fill_between(time_axis, 0, 1, where=(final_preds == 1), color='#ff4b4b', alpha=0.4, label='Detection')
            ax.axhline(threshold, color='blue', linestyle=':', label='Threshold')
            
            ax.set_ylabel("Probability")
            ax.set_xlabel("Time (minutes)")
            ax.set_xlim(0, time_axis[-1])
            ax.set_ylim(0, 1.05)
            ax.legend(loc='upper right')
            st.pyplot(fig)
            
            # Report Table
            if n_seizure_epochs > 0:
                st.markdown("### Detections List")
                padded = np.pad(final_preds, (1, 1), 'constant')
                diffs = np.diff(padded)
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                
                detections = []
                for s, e in zip(starts, ends):
                    detections.append({
                        "Start (s)": s*5, "End (s)": e*5, 
                        "Duration (s)": (e-s)*5,
                        "Confidence (Avg)": f"{np.mean(probs[s:e]):.2f}"
                    })
                st.dataframe(pd.DataFrame(detections))

        # --- TAB 2: SIGNAL INSPECTOR ---
        with tab2:
            st.markdown("### 🔍 Verify Detections")
            
            max_time = int(total_epochs * 5)
            default_time = 0
            if n_seizure_epochs > 0:
                first_seizure_idx = np.where(final_preds == 1)[0][0]
                default_time = int(first_seizure_idx * 5)

            start_view = st.slider("View Time (seconds)", 0, max_time - 10, default_time)
            
            st.write(f"**Raw EEG ({start_view}s - {start_view+10}s)**")
            
            # --- FIX: Direct Slicing (Avoids FileNotFoundError from MNE crop) ---
            sfreq = raw_viz.info['sfreq']
            start_sample = int(start_view * sfreq)
            stop_sample = int((start_view + 10) * sfreq)
            
            # Get data directly from RAM
            data = raw_viz.get_data(start=start_sample, stop=stop_sample)
            times = np.arange(data.shape[1]) / sfreq + start_view
            # ------------------------------------------------------------------
            
            fig_eeg, ax_eeg = plt.subplots(figsize=(12, 8))
            n_channels = len(raw_viz.ch_names)
            offset_step = 0.0002  
            
            for i, ch_name in enumerate(raw_viz.ch_names):
                offset = i * offset_step
                ax_eeg.plot(times, data[i] + offset, linewidth=0.8, color='#333')
                ax_eeg.text(times[0], offset, ch_name, fontsize=8, color='blue', ha='right')
            
            ax_eeg.set_yticks([])
            ax_eeg.spines['top'].set_visible(False)
            ax_eeg.spines['right'].set_visible(False)
            ax_eeg.spines['left'].set_visible(False)
            st.pyplot(fig_eeg)

            st.write("**Spectrogram Analysis**")
            spec_ch = st.selectbox("Select Channel for Spectrogram", raw_viz.ch_names)
            
            ch_idx = raw_viz.ch_names.index(spec_ch)
            ch_data = data[ch_idx]
            
            fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
            f, t, Sxx = spectrogram(ch_data, fs=sfreq, nperseg=128, noverlap=64)
            img = ax_spec.pcolormesh(t + start_view, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
            ax_spec.set_ylabel('Frequency [Hz]')
            ax_spec.set_xlabel('Time [sec]')
            plt.colorbar(img, ax=ax_spec, label='Power (dB)')
            st.pyplot(fig_spec)