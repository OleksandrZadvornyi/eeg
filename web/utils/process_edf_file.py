import mne
import numpy as np
import os
import gudhi as gd
from mne.preprocessing import ICA  # <--- Added Import

# --- Helper Function for ICA ---
def apply_ica_cleaning(raw, n_components=15, random_state=42):
    """
    Fits ICA on the raw object and removes artifact components.
    """
    # 1. Fit ICA
    # We use n_components=15 (must be <= n_channels)
    ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto")
    
    # We fit on the raw data (which is already filtered in the main loop)
    # verbose=False to keep the console clean
    ica.fit(raw, verbose=False)

    # 2. Detect Artifacts (Auto-rejection logic from plotting.py)
    # WARNING: Seizures usually have high amplitude. 
    # Automated rejection based on peak-to-peak amplitude > 200uV might REMOVE seizures.
    # It is safer to use EOG/ECG correlation if you have those channels, 
    # or rely on manual inspection.
    
    # --- AUTOMATIC EXCLUSION LOGIC (Use with caution) ---
    # components = ica.get_components()
    # for idx, component in enumerate(components):
    #     peak_to_peak = component.max() - component.min()
    #     threshold_value = 200e-6 
    #     if peak_to_peak > threshold_value:
    #         ica.exclude.append(idx)
    #         print(f"ICA: Auto-excluding component {idx} (PTP > 200uV)")
    
    # If you want to use the exclusion list, uncomment the lines above.
    # Otherwise, this will just fit ICA and apply it (which might not clean much 
    # without defining exclusions, but prepares the data structure).

    # 3. Apply cleaning
    raw_cleaned = ica.apply(raw, verbose=False)
    return raw_cleaned

# Function from the paper (utils.py)
def compute_persistence_diagram(point_cloud):
    # Create Rips Complex
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=1) 
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence_diagram = simplex_tree.persistence()

    result = []
    for dim, (birth, death) in persistence_diagram:
        # 1. Drop infinite cycles (mandatory!)
        if death == float('inf'):
            continue
            
        # 2. (Optional, but recommended for EEG) Take only H1 (cycles/loops)
        # H0 is just clustering of points. H1 is "holes" in the data.
        # If you want both H0 and H1, comment out the next two lines.
        if dim != 1:
            continue
            
        result.append((dim, (birth, death)))
        
    return result

def compute_landscape_values(diag, grid):
    landscape_values = np.zeros_like(grid)
    
    for interval in diag:
        dim, (birth, death) = interval
        # Here we take only H1 (cycles) or H0 (connected components), depending on dim
        # In the paper the code is simplified, but usually H1 is interesting for EEG
        for i, t in enumerate(grid):
            if birth < t < death:
                landscape_values[i] = max(landscape_values[i], min(t - birth, death - t))
                
    return landscape_values

def process_edf_file(edf_path, seizure_intervals):
    """
    Runs Steps 1-4 on a single .edf file.
    Enforces a standard 18-channel montage with robust mapping.
    """
    
    print(f"\nProcessing {os.path.basename(edf_path)} (Seizures: {len(seizure_intervals)})...")

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
        print("Normalizing channel names...")
        rename_map = {}
        current_chans = raw.ch_names
        
        for ch in current_chans:
            clean_name = ch.upper().strip()
            clean_name = clean_name.replace('-0', '').replace('-REF', '').replace('-Ref', '')

            # Apply 10-20 aliases
            for old, new in CHANNEL_ALIAS.items():
                clean_name = clean_name.replace(old, new)

            if clean_name in STANDARD_CHANNELS:
                rename_map[ch] = clean_name

        # Apply renaming
        if rename_map:
            raw.rename_channels(rename_map)

        # Check for missing channels
        present_channels = raw.ch_names
        missing_chans = [ch for ch in STANDARD_CHANNELS if ch not in present_channels]
        
        if len(missing_chans) > 0:
            print(f"Missing required channels: {missing_chans}. Skipping file.")
            return None, None

        # Pick and reorder channels
        raw.pick_channels(STANDARD_CHANNELS)

        # --- Step 2: Filter and Epoch ---
        print("Filtering data (1.0–50 Hz)...")
        raw.filter(l_freq=1.00, h_freq=50, verbose=False)
        
        # --- NEW STEP: ICA ---
        print("Applying ICA...")
        # We apply ICA on the continuous data before cutting it into epochs
        try:
            raw = apply_ica_cleaning(raw, n_components=15)
        except Exception as e:
            print(f"ICA Failed: {e}. Proceeding with raw data.")
        
        epoch_duration = 5
        epochs = mne.make_fixed_length_epochs(
            raw, 
            duration=epoch_duration, 
            preload=True,
            verbose=False
        )
        
        if len(epochs) == 0:
            print("No epochs extracted. Skipping file.")
            return None, None

        # --- Step 3: Feature Extraction ---
        print("Extracting spectral features...")

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
            for band in bands:
                fmin, fmax = bands[band]
                idx = np.logical_and(freqs >= fmin, freqs < fmax)
                band_power_per_channel = np.mean(epoch_psd[:, idx], axis=1)
                epoch_features.append(band_power_per_channel)
            spectral_features.append(np.concatenate(epoch_features))
        
        spectral_features = np.array(spectral_features)
        spectral_features = 10 * np.log10(spectral_features + 1e-20)
        
        # --- Topological Features ---
        print("Extracting TDA features...")
        
        # Log transform PSD first to handle magnitudes
        psd_log = 10 * np.log10(psd_data + 1e-20)
        
        # Min-Max Normalization per epoch to range [0, 1]
        # This ensures the "shape" is captured independently of signal volume
        p_min = psd_log.min(axis=(1,2), keepdims=True)
        p_max = psd_log.max(axis=(1,2), keepdims=True)
        psd_norm = (psd_log - p_min) / (p_max - p_min + 1e-10)

        # Grid must match the normalized range [0, 1]
        grid = np.linspace(0, 1, 100)
        tda_features_list = []

        for epoch_idx in range(len(psd_data)):
            # Form point cloud: each channel is a point in frequency space
            point_cloud = psd_norm[epoch_idx]

            diag = compute_persistence_diagram(point_cloud)
            landscape = compute_landscape_values(diag, grid)

            tda_features_list.append(landscape)

        tda_features = np.array(tda_features_list)

        all_epoch_data = epochs.get_data() 
        n_epochs, n_channels, _ = all_epoch_data.shape
        line_length = np.sum(np.abs(np.diff(all_epoch_data, axis=2)), axis=2)
        variance = np.var(all_epoch_data, axis=2)

        all_features = np.hstack((spectral_features, tda_features, line_length, variance))

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

        print("Labeling complete.")
        print(f"Seizure epochs: {np.sum(labels)} / {len(labels)}")

        return all_features, labels

    except Exception as e:
        print(f"Error processing {os.path.basename(edf_path)}: {e}")
        return None, None

# all_features, labels, tda_features = process_edf_file("../PhysioNetData_aws/chb24/chb24_06.edf",  [(1229, 1253)])
# all_features, labels, tda_features = process_edf_file("../PhysioNetData_aws/chb12/chb12_11.edf",  [(1085, 1122)])
