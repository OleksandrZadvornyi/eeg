import mne
import numpy as np
import os
import gudhi as gd
from mne.preprocessing import ICA 

def apply_ica_cleaning(raw, n_components=15, random_state=42):
    """
    Fits and applies Independent Component Analysis to remove artifacts.
    """
    ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto")
    ica.fit(raw, verbose=False)

    # Note: Automated rejection logic (e.g., PTP > 200uV) is disabled to 
    # prevent accidental removal of high-amplitude seizure activity.
    
    raw_cleaned = ica.apply(raw, verbose=False)
    return raw_cleaned

def compute_persistence_diagram(point_cloud):
    """
    Computes H1 (cycles) persistence using a Rips Complex.
    """
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=1) 
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence_diagram = simplex_tree.persistence()

    result = []
    for dim, (birth, death) in persistence_diagram:
        # Exclude infinite cycles and focus on H1 (topological loops)
        if death != float('inf') and dim == 1:
            result.append((dim, (birth, death)))
        
    return result

def compute_landscape_values(diag, grid):
    """
    Maps persistence intervals to a persistence landscape vector.
    """
    landscape_values = np.zeros_like(grid)
    for interval in diag:
        dim, (birth, death) = interval
        for i, t in enumerate(grid):
            if birth < t < death:
                landscape_values[i] = max(landscape_values[i], min(t - birth, death - t))
    return landscape_values

def process_edf_file(edf_path, seizure_intervals):
    """
    Main processing pipeline: Normalization, Filtering, ICA, and Feature Extraction.
    """
    print(f"\nProcessing {os.path.basename(edf_path)} (Seizures: {len(seizure_intervals)})...")

    # Target 10-20 system channel configuration
    STANDARD_CHANNELS = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ'
    ]

    # Map legacy CHB-MIT labels to standard 10-20 nomenclature
    CHANNEL_ALIAS = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Normalize channel naming and verify required montage
        rename_map = {}
        for ch in raw.ch_names:
            clean_name = ch.upper().strip().replace('-0', '').replace('-REF', '').replace('-Ref', '')
            for old, new in CHANNEL_ALIAS.items():
                clean_name = clean_name.replace(old, new)
            if clean_name in STANDARD_CHANNELS:
                rename_map[ch] = clean_name

        if rename_map:
            raw.rename_channels(rename_map)

        missing_chans = [ch for ch in STANDARD_CHANNELS if ch not in raw.ch_names]
        if missing_chans:
            print(f"Missing channels: {missing_chans}. Skipping.")
            return None, None

        raw.pick_channels(STANDARD_CHANNELS)

        # Signal Preprocessing
        raw.filter(l_freq=1.00, h_freq=50, verbose=False)
        try:
            raw = apply_ica_cleaning(raw, n_components=15)
        except Exception as e:
            print(f"ICA Failed: {e}. Using raw filtered data.")
        
        epoch_duration = 5
        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True, verbose=False)
        
        if len(epochs) == 0:
            return None, None

        # --- Spectral Feature Extraction ---
        sfreq = epochs.info['sfreq']
        bands = {
            "delta": (0.5, 4), "theta": (4, 8),
            "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)
        }

        psd = epochs.compute_psd(method="welch", n_fft=int(sfreq), fmin=0.5, fmax=50.0, verbose=False)
        psd_data = psd.get_data()
        freqs = psd.freqs

        spectral_features = []
        for epoch_psd in psd_data:
            epoch_features = []
            for band, (fmin, fmax) in bands.items():
                idx = np.logical_and(freqs >= fmin, freqs < fmax)
                epoch_features.append(np.mean(epoch_psd[:, idx], axis=1))
            spectral_features.append(np.concatenate(epoch_features))
        
        spectral_features = 10 * np.log10(np.array(spectral_features) + 1e-20)
        
        # --- Topological Data Analysis (TDA) ---
        # Normalize PSD to [0, 1] to capture scale-invariant topological shapes
        psd_log = 10 * np.log10(psd_data + 1e-20)
        p_min = psd_log.min(axis=(1,2), keepdims=True)
        p_max = psd_log.max(axis=(1,2), keepdims=True)
        psd_norm = (psd_log - p_min) / (p_max - p_min + 1e-10)

        grid = np.linspace(0, 1, 100)
        tda_features_list = []

        for epoch_idx in range(len(psd_data)):
            point_cloud = psd_norm[epoch_idx]
            diag = compute_persistence_diagram(point_cloud)
            landscape = compute_landscape_values(diag, grid)
            tda_features_list.append(landscape)

        # --- Time-Domain and Final Aggregation ---
        all_epoch_data = epochs.get_data() 
        line_length = np.sum(np.abs(np.diff(all_epoch_data, axis=2)), axis=2)
        variance = np.var(all_epoch_data, axis=2)

        all_features = np.hstack((spectral_features, np.array(tda_features_list), line_length, variance))

        # --- Labeling based on Seizure Metadata ---
        labels = np.zeros(len(epochs), dtype=int)
        epoch_start_times = epochs.events[:, 0] / sfreq

        for i in range(len(epochs)):
            e_start, e_end = epoch_start_times[i], epoch_start_times[i] + epoch_duration
            if any(e_start < s_end and e_end > s_start for s_start, s_end in seizure_intervals):
                labels[i] = 1

        print(f"Done. Seizure epochs found: {np.sum(labels)}")
        return all_features, labels

    except Exception as e:
        print(f"Critical error in {os.path.basename(edf_path)}: {e}")
        return None, None