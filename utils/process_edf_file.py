import mne
import numpy as np
import os

def process_edf_file(edf_path, seizure_intervals):
    """
    Runs Steps 1-4 on a single .edf file.
    Enforces a standard 18-channel montage with robust mapping.
    """
    
    # 1. Define the TARGET standard we want (International 10-20)
    STANDARD_CHANNELS = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ'
    ]

    # 2. Define common aliases (Old Terminology -> New Terminology)
    # CHB-MIT mixes these. T3=T7, T4=T8, T5=P7, T6=P8
    CHANNEL_ALIAS = {
        'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
    }

    try:
        # --- Step 1: Load Data ---
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # --- ROBUST CHANNEL NORMALIZATION ---
        
        # A. Create a mapping of current names to clean standard names
        #    Example: "T7-P7-1" -> "T7-P7"
        #    Example: "Fp1-F7"  -> "FP1-F7"
        rename_map = {}
        current_chans = raw.ch_names
        
        for ch in current_chans:
            # Normalize: Upper case, remove spaces
            clean_name = ch.upper().strip()
            
            # Remove common suffixes like "-0", "-REF", etc.
            clean_name = clean_name.replace('-0', '').replace('-REF', '').replace('-Ref', '')
            
            # Apply 10-20 aliases (e.g., replace T3 with T7)
            for old, new in CHANNEL_ALIAS.items():
                clean_name = clean_name.replace(old, new)
            
            # If the cleaned name is in our standard list, map it!
            if clean_name in STANDARD_CHANNELS:
                rename_map[ch] = clean_name

        # B. Apply the renaming
        if rename_map:
            raw.rename_channels(rename_map)

        # C. Check for missing channels
        # Now that we've renamed everything possible, do we have the 18 we need?
        present_channels = raw.ch_names
        missing_chans = [ch for ch in STANDARD_CHANNELS if ch not in present_channels]
        
        if len(missing_chans) > 0:
            # PRINT THE WARNING so we know why it failed
            # print(f"Skipping {os.path.basename(edf_path)}: Missing {missing_chans}") 
            return None, None

        # D. Pick and Reorder
        # This keeps strictly the 18 channels in the correct order
        raw.pick_channels(STANDARD_CHANNELS)
        
        print(f"Processing {os.path.basename(edf_path)} (Seizures: {len(seizure_intervals)})...")

        # --- Step 2: Filter and Epoch ---
        raw.filter(l_freq=0.5, h_freq=50, verbose=False)
        
        epoch_duration = 5
        epochs = mne.make_fixed_length_epochs(
            raw, 
            duration=epoch_duration, 
            preload=True,
            verbose=False
        )
        
        if len(epochs) == 0:
            return None, None

        # --- Step 3: Feature Extraction ---
        
        # Part A: Spectral
        sfreq = epochs.info['sfreq']
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), 
                 "beta": (13, 30), "gamma": (30, 50)}
        psd = epochs.compute_psd(method="welch", n_fft=int(sfreq), fmin=0.5, fmax=50.0, verbose=False)
        psd_data = psd.get_data()
        freqs = psd.freqs
        
        spectral_features = []
        for epoch_psd in psd_data:
            epoch_features = []
            for band in bands:
                fmin, fmax = bands[band]
                idx = np.logical_and(freqs >= fmin, freqs < fmax)
                band_power_per_channel = np.mean(epoch_psd[:, idx], axis=1)
                epoch_features.append(band_power_per_channel)
            spectral_features.append(np.concatenate(epoch_features))
        spectral_features = np.array(spectral_features)

        # Part B: Topological
        all_epoch_data = epochs.get_data() 
        n_epochs, n_channels, _ = all_epoch_data.shape
        triu_indices = np.triu_indices(n_channels, k=1)
        topological_features_list = []
        for i in range(n_epochs):
            epoch_data = all_epoch_data[i]
            corr_matrix = np.corrcoef(epoch_data)
            epoch_conn_features = corr_matrix[triu_indices]
            topological_features_list.append(epoch_conn_features)
        topological_features = np.array(topological_features_list)

        # Part B.5: Time-Domain
        line_length = np.sum(np.abs(np.diff(all_epoch_data, axis=2)), axis=2)
        variance = np.var(all_epoch_data, axis=2)
        
        # Part C: Combine
        all_features = np.hstack((spectral_features, topological_features, line_length, variance))

        # --- Step 4: Labeling ---
        labels = np.zeros(n_epochs, dtype=int)
        epoch_start_times = epochs.events[:, 0] / epochs.info['sfreq']

        for i in range(n_epochs):
            epoch_start = epoch_start_times[i]
            epoch_end = epoch_start + epoch_duration
            for seizure_start, seizure_end in seizure_intervals:
                if (epoch_start < seizure_end) and (epoch_end > seizure_start):
                    labels[i] = 1
                    break 
                    
        return all_features, labels

    except Exception as e:
        print(f"Error processing {os.path.basename(edf_path)}: {e}")
        return None, None