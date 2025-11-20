import mne
import numpy as np

# Copy the exact channel config from your training script
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
    'FZ-CZ', 'CZ-PZ'
]
CHANNEL_ALIAS = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}

def extract_features_for_inference(edf_path):
    """
    Processes an EDF file and returns X (features) and the time indices.
    Does NOT require ground truth seizure intervals.
    """
    try:
        # 1. Load
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # 2. Rename & Standardize Channels (Your logic)
        rename_map = {}
        for ch in raw.ch_names:
            clean_name = ch.upper().strip().replace('-0', '').replace('-REF', '').replace('-Ref', '')
            for old, new in CHANNEL_ALIAS.items():
                clean_name = clean_name.replace(old, new)
            if clean_name in STANDARD_CHANNELS:
                rename_map[ch] = clean_name
        
        if rename_map:
            raw.rename_channels(rename_map)

        # Check if we have all channels
        if not all(ch in raw.ch_names for ch in STANDARD_CHANNELS):
            return None, "Missing required EEG channels."

        raw.pick_channels(STANDARD_CHANNELS)
        
        # 3. Filter & Epoch
        raw.filter(l_freq=0.5, h_freq=50, verbose=False)
        epoch_duration = 5
        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True, verbose=False)
        
        if len(epochs) == 0:
            return None, "File too short."

        # 4. Feature Extraction (Exact copy of your training logic)
        
        # A. Spectral
        sfreq = epochs.info['sfreq']
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
        psd = epochs.compute_psd(method="welch", n_fft=int(sfreq), fmin=0.5, fmax=50.0, verbose=False)
        psd_data = psd.get_data()
        freqs = psd.freqs
        
        spectral_features = []
        for epoch_psd in psd_data:
            epoch_features = []
            for band in bands:
                fmin, fmax = bands[band]
                idx = np.logical_and(freqs >= fmin, freqs < fmax)
                epoch_features.append(np.mean(epoch_psd[:, idx], axis=1))
            spectral_features.append(np.concatenate(epoch_features))
        spectral_features = np.array(spectral_features)

        # B. Topological
        all_epoch_data = epochs.get_data()
        n_epochs, n_channels, _ = all_epoch_data.shape
        triu_indices = np.triu_indices(n_channels, k=1)
        
        topo_list = []
        for i in range(n_epochs):
            corr_matrix = np.corrcoef(all_epoch_data[i])
            topo_list.append(corr_matrix[triu_indices])
        topological_features = np.array(topo_list)

        # B.5 Time Domain (Z-Score Normalized per epoch)
        means = np.mean(all_epoch_data, axis=2, keepdims=True)
        stds = np.std(all_epoch_data, axis=2, keepdims=True)
        stds[stds == 0] = 1 
        normalized_data = (all_epoch_data - means) / stds
        
        line_length = np.sum(np.abs(np.diff(normalized_data, axis=2)), axis=2)
        variance = np.var(normalized_data, axis=2)
        
        # Combine
        X = np.hstack((spectral_features, topological_features, line_length, variance))
        
        # Get timestamps for display
        timestamps = epochs.events[:, 0] / epochs.info['sfreq']
        
        return (X, timestamps), None

    except Exception as e:
        return None, str(e)