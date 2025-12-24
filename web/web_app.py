import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import tempfile
import matplotlib.pyplot as plt
from scipy.signal import medfilt, spectrogram
import mne

# Internal pipeline imports
try:
    from utils.eeg_processor import EEGPipeline, EEGConfig
except ImportError:
    st.error("Critical Error: Could not import EEGPipeline dependencies.")

class PrivacyManager:
    @staticmethod
    def anonymize_raw(raw):
        """Removes identifying subject metadata from the MNE object."""
        if raw.info['subject_info'] is not None:
            raw.info['subject_info'] = None
        return raw

class EEGVisualizer:
    @staticmethod
    def plot_timeline(probs, preds, threshold, epoch_duration=5):
        """Generates a probability and detection timeline plot."""
        total_epochs = len(probs)
        time_axis = np.arange(total_epochs) * epoch_duration / 60 
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(time_axis, probs, color='#333333', alpha=0.6, linewidth=1, label='Probability')
        ax.fill_between(time_axis, 0, 1, where=(preds == 1), color='#ff4b4b', alpha=0.4, label='Detection')
        ax.axhline(threshold, color='blue', linestyle=':', label='Threshold')
        
        ax.set_ylabel("Probability")
        ax.set_xlabel("Time (minutes)")
        ax.set_xlim(0, time_axis[-1])
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right')
        return fig

    @staticmethod
    def plot_raw_signal(raw, start_time, duration=10):
        """Plots multi-channel EEG traces for a specific time window."""
        sfreq = raw.info['sfreq']
        start_sample = int(start_time * sfreq)
        stop_sample = int((start_time + duration) * sfreq)
        
        data = raw.get_data(start=start_sample, stop=stop_sample)
        times = np.arange(data.shape[1]) / sfreq + start_time
        
        fig, ax = plt.subplots(figsize=(12, 8))
        offset_step = 0.0002  # Vertical separation between channels
        
        for i, ch_name in enumerate(raw.ch_names):
            offset = i * offset_step
            ax.plot(times, data[i] + offset, linewidth=0.8, color='#333')
            ax.text(times[0], offset, ch_name, fontsize=8, color='blue', ha='right')
        
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        return fig

    @staticmethod
    def plot_spectrogram(data, sfreq, start_time):
        """Generates a power spectral density heatmap."""
        fig, ax = plt.subplots(figsize=(10, 4))
        f, t, Sxx = spectrogram(data, fs=sfreq, nperseg=128, noverlap=64)
        img = ax.pcolormesh(t + start_time, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        plt.colorbar(img, ax=ax, label='Power (dB)')
        return fig

class ModelHandler:
    """Manages model loading and inference."""
    def __init__(self, model_path='seizure_model_final.pkl', config_path='model_config.json'):
        self.model = None
        self.config = {}
        self._load(model_path, config_path)

    def _load(self, model_path, config_path):
        try:
            self.model = joblib.load(model_path)
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            pass 

    def is_loaded(self):
        return self.model is not None

    def predict(self, features):
        if not self.is_loaded(): return None
        return self.model.predict_proba(features)[:, 1]

class SeizureApp:
    def __init__(self):
        st.set_page_config(page_title="EEG Seizure Detector", page_icon="🧠", layout="wide")
        self.pipeline = EEGPipeline() 
        self.model_handler = ModelHandler()
        self._init_session_state()

    def _init_session_state(self):
        """Initializes Streamlit state variables to persist data across reruns."""
        if 'file_name' not in st.session_state: st.session_state.file_name = None
        if 'probs' not in st.session_state: st.session_state.probs = None
        if 'raw_viz' not in st.session_state: st.session_state.raw_viz = None
        if 'features_extracted' not in st.session_state: st.session_state.features_extracted = False

    def render_sidebar(self):
        st.sidebar.header("Configuration")
        
        if not self.model_handler.is_loaded():
            st.sidebar.error("Model files not found.")
            st.stop()

        st.sidebar.success("Model Loaded")
        
        # Hyperparameters for post-processing
        default_thresh = self.model_handler.config.get('best_threshold', 0.5)
        thresh = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, default_thresh)
        
        default_kernel = self.model_handler.config.get('filter_kernel', 5)
        kernel = st.sidebar.slider("Smoothing Kernel", 1, 15, default_kernel, step=2)

        st.sidebar.markdown("---")
        with st.sidebar.expander("Privacy & Disclaimer"):
            st.warning("Not a Medical Device.")
            st.info("Data processed locally. No cloud uploads.")
            
        return thresh, kernel

    def process_file(self, uploaded_file):
        """Handles file upload, feature extraction, and prediction."""
        if st.session_state.file_name == uploaded_file.name:
            return

        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Extract features and generate seizure probabilities
                features, _ = self.pipeline.process(tmp_path, seizure_intervals=[])
                if features is None:
                    st.error("Feature extraction failed.")
                    st.stop()

                probs = self.model_handler.predict(features)

                # Prepare raw object for inspector visualization
                raw_viz = self.pipeline.preprocessor.load_and_clean(tmp_path)
                raw_viz = PrivacyManager.anonymize_raw(raw_viz)

                st.session_state.probs = probs
                st.session_state.raw_viz = raw_viz
                st.session_state.file_name = uploaded_file.name
                st.session_state.features_extracted = True

            except Exception as e:
                st.error(f"Processing Error: {e}")
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

    def render_results(self, threshold, kernel):
        if not st.session_state.features_extracted:
            return

        probs = st.session_state.probs
        raw = st.session_state.raw_viz
        
        # Apply threshold and median filtering for temporal smoothing
        preds = medfilt((probs >= threshold).astype(int), kernel_size=kernel)
        n_seizures = np.sum(preds)

        tab1, tab2 = st.tabs(["📊 Detection Results", "🔍 Signal Inspector"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Duration", f"{(len(preds)*5)/60:.1f} min")
            c2.metric("Seizure Burden", f"{n_seizures*5} sec")
            if n_seizures > 0:
                c3.error("⚠️ Seizure Detected")
            else:
                c3.success("✅ Normal Record")
            
            st.pyplot(EEGVisualizer.plot_timeline(probs, preds, threshold))
            
            if n_seizures > 0:
                self._render_detection_table(preds, probs)

        with tab2:
            self._render_inspector(raw, preds, n_seizures)

    def _render_detection_table(self, preds, probs):
        """Parses prediction blocks into a readable start/end list."""
        st.markdown("### Detections List")
        padded = np.pad(preds, (1, 1), 'constant')
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        detections = []
        for s, e in zip(starts, ends):
            detections.append({
                "Start (s)": s*5, "End (s)": e*5, 
                "Confidence": f"{np.mean(probs[s:e]):.2f}"
            })
        st.dataframe(pd.DataFrame(detections))

    def _render_inspector(self, raw, preds, n_seizures):
        """Interactive tool to view raw signals and spectrograms."""
        max_time = int(len(preds) * 5)
        # Default view to first seizure event if present
        default = int(np.where(preds == 1)[0][0] * 5) if n_seizures > 0 else 0
        
        start_view = st.slider("Jump to Time (s)", 0, max_time - 10, default)
        
        st.pyplot(EEGVisualizer.plot_raw_signal(raw, start_view))
        
        spec_ch = st.selectbox("Select Spectrogram Channel", raw.ch_names)
        ch_idx = raw.ch_names.index(spec_ch)
        
        sfreq = raw.info['sfreq']
        start_samp = int(start_view * sfreq)
        end_samp = int((start_view + 10) * sfreq)
        data_segment = raw.get_data(start=start_samp, stop=end_samp)[ch_idx]
        
        st.pyplot(EEGVisualizer.plot_spectrogram(data_segment, sfreq, start_view))

    def run(self):
        st.title("🧠 EEG Seizure Detection System")
        thresh, kernel = self.render_sidebar()
        
        uploaded_file = st.file_uploader("Upload EEG Record (.edf)", type=['edf'])
        if uploaded_file:
            self.process_file(uploaded_file)
            self.render_results(thresh, kernel)

if __name__ == "__main__":
    app = SeizureApp()
    app.run()