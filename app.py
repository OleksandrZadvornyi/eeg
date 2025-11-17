import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


########################################################
########### Step 1: Load Data and Preprocess ###########
########################################################

raw = mne.io.read_raw_edf('chb01_03.edf', preload=True)
raw.filter(l_freq=0.5, h_freq=50)



########################################################
########### Step 2: Create Epochs (Windows) ############
########################################################

# Set the epoch duration in seconds
epoch_duration = 5  # (e.g., 5, 10, 30 seconds)

# Create fixed-length epochs from the filtered data
epochs = mne.make_fixed_length_epochs(
    raw, 
    duration=epoch_duration, 
    preload=True
)



########################################################
###### Step 3: Extract Features (For Each Epoch) #######
########################################################

# --- Part A: Spectral Features (Power in Bands) ---

print("Starting Step 3a: Extracting spectral features...")

# Define the frequency bands we care about
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50) 
}

# Get the sampling frequency from your data
sfreq = epochs.info['sfreq']

# Compute the Power Spectral Density (PSD) for every epoch
# This calculates the power of all frequencies
psd = epochs.compute_psd(method="welch", n_fft=int(sfreq), fmin=0.5, fmax=50.0)
psd_data = psd.get_data()  # Shape is (n_epochs, n_channels, n_freqs)
freqs = psd.freqs

# 

# Calculate the average power in each band for each epoch
spectral_features = []

for epoch_psd in psd_data:
    epoch_features = []
    for band in bands:
        fmin, fmax = bands[band]
        # Find frequency indices for this band
        idx = np.logical_and(freqs >= fmin, freqs < fmax)
        # Calculate mean power for this band across all channels
        # .mean(axis=1) averages across the frequencies in the band
        band_power_per_channel = np.mean(epoch_psd[:, idx], axis=1)
        epoch_features.append(band_power_per_channel)
    
    # Flatten features for this epoch (n_channels * n_bands)
    spectral_features.append(np.concatenate(epoch_features))

# Convert list to a numpy array
spectral_features = np.array(spectral_features)
# This array now has shape (n_epochs, 115)  (23 channels * 5 bands)


# --- Part B: Topological Features (Connectivity) ---

print("Starting Step 3b: Extracting topological (connectivity) features...")

# Get all epoch data at once
# Shape is (n_epochs, n_channels, n_time_samples)
all_epoch_data = epochs.get_data() 
n_epochs, n_channels, _ = all_epoch_data.shape

# We only need the upper triangle of the matrix (to avoid duplicate features)
triu_indices = np.triu_indices(n_channels, k=1) # k=1 excludes the diagonal

# Create a list to store the features for each epoch
topological_features_list = []

# Loop over each epoch to calculate its correlation matrix
for i in range(n_epochs):
    # Get the data for this single epoch (shape: n_channels, n_time_samples)
    epoch_data = all_epoch_data[i]
    
    # Calculate the Pearson correlation matrix for this epoch
    # np.corrcoef returns a matrix of shape (n_channels, n_channels)
    corr_matrix = np.corrcoef(epoch_data)
    
    # 

    # Extract the upper triangle values from the matrix
    epoch_conn_features = corr_matrix[triu_indices]
    
    # Add this epoch's features to our list
    topological_features_list.append(epoch_conn_features)

# Convert the list of feature vectors into a 2D numpy array
topological_features = np.array(topological_features_list)

# This array now has shape (n_epochs, 253) ( (23 * 22) / 2 features)


# --- Part C: Combine All Features ---

print("Starting Step 3c: Combining features...")

# Stack the two feature sets side-by-side
# Shape (n_epochs, 115) + (n_epochs, 253) -> (n_epochs, 368)
all_features = np.hstack((spectral_features, topological_features))

print("\n--- Feature Extraction Complete ---")
print(f"Spectral features shape:    {spectral_features.shape}")
print(f"Topological features shape: {topological_features.shape}")
print(f"Combined features shape:    {all_features.shape}")



########################################################
####### Step 4: Create Feature Vector and Labels #######
########################################################


# --- Part A: Define Seizure Times (Manual Step) ---

# We know these from our previous steps
n_epochs = len(all_features)
epoch_duration = 5  # This must be the same value from Step 2

# ====> 1. FILL THIS IN <====
# Look in your 'chb01-summary.txt' for 'chb01_01.edf'.
# Add the (start, end) times for *every* seizure in that file.
# If the file has no seizures, just leave it as an empty list: []
#
# Example from chb03_01.edf: seizure_intervals = [ (362, 414) ]
# Example for a file with 2 seizures: seizure_intervals = [ (100, 120), (500, 530) ]
#
seizure_intervals = [
    (2996, 3036)
    # (seizure_start_seconds, seizure_end_seconds),
    # (seizure_2_start, seizure_2_end),
]


# --- Part B: Get Epoch Times ---

# Initialize all labels to 0 (non-seizure)
labels = np.zeros(n_epochs, dtype=int)

# Get the start time (in seconds) for every epoch
epoch_start_times = epochs.events[:, 0] / epochs.info['sfreq']


# --- Part C: Create Labels ---

print(f"Creating labels for {n_epochs} epochs...")

# Loop through every epoch to see if it overlaps with a seizure
for i in range(n_epochs):
    epoch_start = epoch_start_times[i]
    epoch_end = epoch_start + epoch_duration

    # Check this epoch against all known seizure intervals
    for seizure_start, seizure_end in seizure_intervals:
        
        # Check for overlap
        if (epoch_start < seizure_end) and (epoch_end > seizure_start):
            labels[i] = 1  # Label this epoch as "seizure"
            break # This epoch is labeled, no need to check other intervals

# --- Summary ---
print("Labeling complete.")
print(f"Total epochs: {n_epochs}")
print(f"Seizure epochs found (label 1): {np.sum(labels)}")
print(f"Non-seizure epochs (label 0): {n_epochs - np.sum(labels)}")



########################################################
############### Step 5: Train the model ################
########################################################

# --- Part A: Split Your Data ---
# This is a TEMPORARY split for our single-file test.
# 'test_size=0.2' means we hold back 20% of the data for testing.
# 'random_state=42' just ensures you get the same 'random' split I do.

X_train, X_test, y_train, y_test = train_test_split(
    all_features, 
    labels, 
    test_size=0.2, 
    random_state=42,
    stratify=labels  # Ensures 1s and 0s are split proportionally
)

print(f"Total samples: {len(all_features)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --- Part B: Define and Train the Model ---

print("\nTraining the model...")

# Define the model.
# class_weight="balanced" is THE MOST IMPORTANT parameter.
# It tells the model to pay extra attention to the 'seizure' (1) class,
# which is rare.
model = RandomForestClassifier(
    n_estimators=100,         # 100 "trees" in the forest
    class_weight="balanced",  # CRITICAL for imbalanced data
    random_state=42
)

# Train the model!
# This is where the model "learns" from your features and labels.
model.fit(X_train, y_train)


print("\n--- Model Training Complete ---")



########################################################
######### Step 6: Evaluate Model Performance ###########
########################################################



# --- Part A: Make Predictions ---

print("Making predictions on the test data...")
# Use the trained model to predict labels for the unseen test data
predictions = model.predict(X_test)


# --- Part B: Show the Results ---

print("\n--- Classification Report ---")
# This report is the most important output.
# It breaks down performance for each class (0 and 1).
report = classification_report(y_test, predictions, target_names=["Non-Seizure (0)", "Seizure (1)"])
print(report)


# --- Part C: Visualize the Results (Optional but helpful) ---

print("\n--- Confusion Matrix ---")
# A confusion matrix shows *how* the model was wrong.
cm = confusion_matrix(y_test, predictions)
print(cm)

# 

# You can use this to create a heatmap visualization
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Non-Seizure', 'Predicted Seizure'], 
            yticklabels=['Actual Non-Seizure', 'Actual Seizure'])
plt.title('Confusion Matrix')
plt.show()