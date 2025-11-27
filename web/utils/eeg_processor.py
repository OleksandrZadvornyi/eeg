import mne
import numpy as np
import os
import gudhi as gd
from mne.preprocessing import ICA
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# --- Configuration Class ---
@dataclass
class EEGConfig:
    """Holds constant configuration for EEG processing."""
    standard_channels: List[str] = field(default_factory=lambda: [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ'
    ])
    
    channel_alias: Dict[str, str] = field(default_factory=lambda: {
        'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
    })
    
    freq_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (0.5, 4), "theta": (4, 8),
        "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)
    })
    
    epoch_duration: int = 5
    sfreq_min: float = 1.0
    sfreq_max: float = 50.0

# --- Topological Analysis Class ---
class TDAExtractor:
    """Handles Topological Data Analysis (Persistent Homology)."""
    
    def __init__(self, grid_size: int = 100):
        self.grid = np.linspace(0, 1, grid_size)

    def _compute_persistence_diagram(self, point_cloud):
        rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=1) 
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence_diagram = simplex_tree.persistence()

        result = []
        for dim, (birth, death) in persistence_diagram:
            if death == float('inf'): continue
            if dim != 1: continue # Only H1 (cycles)
            result.append((dim, (birth, death)))
        return result

    def _compute_landscape_values(self, diag):
        landscape_values = np.zeros_like(self.grid)
        for _, (birth, death) in diag:
            for i, t in enumerate(self.grid):
                if birth < t < death:
                    landscape_values[i] = max(landscape_values[i], min(t - birth, death - t))
        return landscape_values

    def extract(self, psd_data: np.ndarray) -> np.ndarray:
        """
        Extracts TDA features from PSD data.
        psd_data shape: (n_epochs, n_channels, n_freqs)
        """
        # Log transform and Normalize
        psd_log = 10 * np.log10(psd_data + 1e-20)
        p_min = psd_log.min(axis=(1, 2), keepdims=True)
        p_max = psd_log.max(axis=(1, 2), keepdims=True)
        psd_norm = (psd_log - p_min) / (p_max - p_min + 1e-10)

        features_list = []
        for epoch_idx in range(len(psd_norm)):
            point_cloud = psd_norm[epoch_idx]
            diag = self._compute_persistence_diagram(point_cloud)
            landscape = self._compute_landscape_values(diag)
            features_list.append(landscape)
            
        return np.array(features_list)

# --- Preprocessing Class ---
class EEGPreprocessor:
    """Handles loading, cleaning, channel standardization, and ICA."""
    
    def __init__(self, config: EEGConfig):
        self.config = config
        self.ica = ICA(n_components=15, random_state=42, max_iter="auto")

    def _standardize_channels(self, raw: mne.io.Raw) -> Optional[mne.io.Raw]:
        rename_map = {}
        current_chans = raw.ch_names
        
        for ch in current_chans:
            clean_name = ch.upper().strip().replace('-0', '').replace('-REF', '').replace('-Ref', '')
            for old, new in self.config.channel_alias.items():
                clean_name = clean_name.replace(old, new)

            if clean_name in self.config.standard_channels:
                rename_map[ch] = clean_name

        if rename_map:
            raw.rename_channels(rename_map)

        # Check completeness
        missing = [ch for ch in self.config.standard_channels if ch not in raw.ch_names]
        if missing:
            print(f"Missing channels: {missing}")
            return None
        
        raw.pick_channels(self.config.standard_channels)
        return raw

    def apply_ica(self, raw: mne.io.Raw) -> mne.io.Raw:
        try:
            self.ica.fit(raw, verbose=False)
            return self.ica.apply(raw, verbose=False)
        except Exception as e:
            print(f"ICA Failed: {e}. Returning raw.")
            return raw

    def load_and_clean(self, file_path: str) -> Optional[mne.io.Raw]:
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            raw = self._standardize_channels(raw)
            if raw is None: return None
            
            raw.filter(l_freq=self.config.sfreq_min, h_freq=self.config.sfreq_max, verbose=False)
            raw = self.apply_ica(raw)
            return raw
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

# --- Main Pipeline Class ---
class EEGPipeline:
    """Orchestrates the conversion of EDF to Features."""
    
    def __init__(self):
        self.config = EEGConfig()
        self.preprocessor = EEGPreprocessor(self.config)
        self.tda_extractor = TDAExtractor()

    def _extract_spectral(self, epochs: mne.Epochs) -> Tuple[np.ndarray, np.ndarray]:
        sfreq = epochs.info['sfreq']
        psd = epochs.compute_psd(method="welch", n_fft=int(sfreq), fmin=0.5, fmax=50.0, verbose=False)
        psd_data = psd.get_data()
        freqs = psd.freqs
        
        spectral_features = []
        for epoch_psd in psd_data:
            epoch_feats = []
            for _, (fmin, fmax) in self.config.freq_bands.items():
                idx = np.logical_and(freqs >= fmin, freqs < fmax)
                band_power = np.mean(epoch_psd[:, idx], axis=1)
                epoch_feats.append(band_power)
            spectral_features.append(np.concatenate(epoch_feats))
            
        spectral_features = 10 * np.log10(np.array(spectral_features) + 1e-20)
        return spectral_features, psd_data

    def _generate_labels(self, epochs: mne.Epochs, seizure_intervals: List[Tuple[float, float]]) -> np.ndarray:
        n_epochs = len(epochs)
        labels = np.zeros(n_epochs, dtype=int)
        
        if not seizure_intervals:
            return labels

        starts = epochs.events[:, 0] / epochs.info['sfreq']
        duration = self.config.epoch_duration

        for i in range(n_epochs):
            t_start = starts[i]
            t_end = t_start + duration
            for s_start, s_end in seizure_intervals:
                if (t_start < s_end) and (t_end > s_start):
                    labels[i] = 1
                    break
        return labels

    def process(self, edf_path: str, seizure_intervals: List[Tuple[float, float]] = None):
        print(f"Processing {os.path.basename(edf_path)}...")
        
        # 1. Load & Clean
        raw = self.preprocessor.load_and_clean(edf_path)
        if raw is None: return None, None

        # 2. Epoch
        epochs = mne.make_fixed_length_epochs(
            raw, duration=self.config.epoch_duration, preload=True, verbose=False
        )
        if len(epochs) == 0: return None, None

        # 3. Spectral Features
        spec_feats, psd_data = self._extract_spectral(epochs)

        # 4. Topological Features
        tda_feats = self.tda_extractor.extract(psd_data)

        # 5. Time-domain stats
        epoch_data = epochs.get_data()
        line_length = np.sum(np.abs(np.diff(epoch_data, axis=2)), axis=2)
        variance = np.var(epoch_data, axis=2)

        # 6. Combine
        all_features = np.hstack((spec_feats, tda_feats, line_length, variance))

        # 7. Labels
        labels = self._generate_labels(epochs, seizure_intervals or [])
        
        return all_features, labels

# Wrapper function to maintain backward compatibility if needed
def process_edf_file(edf_path, seizure_intervals):
    pipeline = EEGPipeline()
    return pipeline.process(edf_path, seizure_intervals)