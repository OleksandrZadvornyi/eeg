import mne
import numpy as np
import os
import gudhi as gd
from mne.preprocessing import ICA
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

@dataclass
class EEGConfig:
    """Encapsulates hardware and signal processing parameters."""
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

class TDAExtractor:
    """Computes Persistent Homology (H1) features from spectral data."""
    
    def __init__(self, grid_size: int = 100):
        self.grid = np.linspace(0, 1, grid_size)

    def _compute_persistence_diagram(self, point_cloud):
        """Generates Rips Complex and extracts finite 1D cycles."""
        rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=1) 
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence_diagram = simplex_tree.persistence()

        result = []
        for dim, (birth, death) in persistence_diagram:
            if death == float('inf') or dim != 1: 
                continue 
            result.append((dim, (birth, death)))
        return result

    def _compute_landscape_values(self, diag):
        """Calculates the persistence landscape vector for the given diagram."""
        landscape_values = np.zeros_like(self.grid)
        for _, (birth, death) in diag:
            for i, t in enumerate(self.grid):
                if birth < t < death:
                    landscape_values[i] = max(landscape_values[i], min(t - birth, death - t))
        return landscape_values

    def extract(self, psd_data: np.ndarray) -> np.ndarray:
        """Transforms raw PSD into normalized topological landscapes."""
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

class EEGPreprocessor:
    """Handles signal cleaning, montage standardization, and artifact removal."""
    
    def __init__(self, config: EEGConfig):
        self.config = config
        self.ica = ICA(n_components=15, random_state=42, max_iter="auto")

    def _standardize_channels(self, raw: mne.io.Raw) -> Optional[mne.io.Raw]:
        """Normalizes channel nomenclature and enforces standard montage."""
        rename_map = {}
        for ch in raw.ch_names:
            clean_name = ch.upper().strip().replace('-0', '').replace('-REF', '').replace('-Ref', '')
            for old, new in self.config.channel_alias.items():
                clean_name = clean_name.replace(old, new)
            if clean_name in self.config.standard_channels:
                rename_map[ch] = clean_name

        if rename_map:
            raw.rename_channels(rename_map)

        missing = [ch for ch in self.config.standard_channels if ch not in raw.ch_names]
        if missing:
            return None
        
        raw.pick_channels(self.config.standard_channels)
        return raw

    def apply_ica(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Fits and applies ICA to minimize non-neural artifacts."""
        try:
            self.ica.fit(raw, verbose=False)
            return self.ica.apply(raw, verbose=False)
        except Exception as e:
            return raw

    def load_and_clean(self, file_path: str) -> Optional[mne.io.Raw]:
        """Pipeline to load EDF, apply montage, filter, and run ICA."""
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            raw = self._standardize_channels(raw)
            if raw is None: return None
            
            raw.filter(l_freq=self.config.sfreq_min, h_freq=self.config.sfreq_max, verbose=False)
            raw = self.apply_ica(raw)
            return raw
        except Exception:
            return None

class EEGPipeline:
    """Orchestrates feature extraction and labeling for EEG segments."""
    
    def __init__(self):
        self.config = EEGConfig()
        self.preprocessor = EEGPreprocessor(self.config)
        self.tda_extractor = TDAExtractor()

    def _extract_spectral(self, epochs: mne.Epochs) -> Tuple[np.ndarray, np.ndarray]:
        """Computes log-power spectral density across defined frequency bands."""
        sfreq = epochs.info['sfreq']
        psd = epochs.compute_psd(method="welch", n_fft=int(sfreq), fmin=0.5, fmax=50.0, verbose=False)
        psd_data = psd.get_data()
        freqs = psd.freqs
        
        spectral_features = []
        for epoch_psd in psd_data:
            epoch_feats = []
            for _, (fmin, fmax) in self.config.freq_bands.items():
                idx = np.logical_and(freqs >= fmin, freqs < fmax)
                epoch_feats.append(np.mean(epoch_psd[:, idx], axis=1))
            spectral_features.append(np.concatenate(epoch_feats))
            
        return 10 * np.log10(np.array(spectral_features) + 1e-20), psd_data

    def _generate_labels(self, epochs: mne.Epochs, seizure_intervals: List[Tuple[float, float]]) -> np.ndarray:
        """Assigns binary labels based on overlap with seizure timestamps."""
        n_epochs = len(epochs)
        labels = np.zeros(n_epochs, dtype=int)
        if not seizure_intervals:
            return labels

        starts = epochs.events[:, 0] / epochs.info['sfreq']
        for i in range(n_epochs):
            t_start, t_end = starts[i], starts[i] + self.config.epoch_duration
            if any(t_start < s_end and t_end > s_start for s_start, s_end in seizure_intervals):
                labels[i] = 1
        return labels

    def process(self, edf_path: str, seizure_intervals: List[Tuple[float, float]] = None):
        """Executes full processing loop from file to multi-domain feature set."""
        raw = self.preprocessor.load_and_clean(edf_path)
        if raw is None: return None, None

        epochs = mne.make_fixed_length_epochs(raw, duration=self.config.epoch_duration, preload=True, verbose=False)
        if len(epochs) == 0: return None, None

        spec_feats, psd_data = self._extract_spectral(epochs)
        tda_feats = self.tda_extractor.extract(psd_data)
        
        # Calculate time-domain metrics
        epoch_data = epochs.get_data()
        line_length = np.sum(np.abs(np.diff(epoch_data, axis=2)), axis=2)
        variance = np.var(epoch_data, axis=2)

        all_features = np.hstack((spec_feats, tda_feats, line_length, variance))
        labels = self._generate_labels(epochs, seizure_intervals or [])
        
        return all_features, labels

def process_edf_file(edf_path, seizure_intervals):
    """Main entry point for EDF file processing."""
    return EEGPipeline().process(edf_path, seizure_intervals)