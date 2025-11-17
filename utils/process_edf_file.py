import mne
import numpy as np
import os

# raw = mne.io.read_raw_edf("./PhysioNetData_aws/chb21/chb21_21.edf", preload=True, exclude=['--0', '--1', '--2', '--3', '--4'], verbose=False)
# dummy_channels = [ch for ch in raw.ch_names if ch.startswith('-')]
# if dummy_channels:
#     print(f"Info for chb21_21.edf: Found and dropping {len(dummy_channels)} dummy channels.")
#     raw.drop_channels(dummy_channels, on_missing='ignore')
# print(len(raw.ch_names))

def process_edf_file(edf_path, seizure_intervals):
    """
    Runs Steps 1-4 on a single .edf file.
    
    Returns a tuple: (all_features, labels) for this file.
    """
    try:
        # --- Step 1 & 2: Load, Filter, Epoch ---
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # 1. Find all channel names that start with a dash
        dummy_channels = [ch for ch in raw.ch_names if ch.startswith('-')]
        
        # 2. Drop them from the raw object
        if dummy_channels:
            print(f"Info for {os.path.basename(edf_path)}: Found and dropping {len(dummy_channels)} dummy channels.")
            raw.drop_channels(dummy_channels, on_missing='ignore')
        
        # Now, AFTER dropping dummies, we check if exactly 23 EEG channels remain.
        # This is the correct way to standardize our data.
        if len(raw.ch_names) != 23:
            print(f"Skipping {os.path.basename(edf_path)}: Has {len(raw.ch_names)} non-dummy channels, not 23.")
            return None, None
        
        print(f"Processing {os.path.basename(edf_path)} (Seizures: {len(seizure_intervals)})...")
                
        raw.filter(l_freq=0.5, h_freq=50, verbose=False)
        
        epoch_duration = 5
        epochs = mne.make_fixed_length_epochs(
            raw, 
            duration=epoch_duration, 
            preload=True,
            verbose=False
        )
        
        if len(epochs) == 0:
            print(f"Skipping {os.path.basename(edf_path)}: No epochs created.")
            return None, None

        # --- Step 3: Feature Extraction ---
        # (This is your exact code from Step 3A and 3B)
        
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
        
        # Part C: Combine
        all_features = np.hstack((spectral_features, topological_features))

        # --- Step 4: Labeling ---
        # (This is your exact code from Step 4B and 4C)
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