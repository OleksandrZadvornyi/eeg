import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
import os  
import glob
import warnings
import sys
import joblib 
from scipy.signal import medfilt
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

from utils.parse_summary_file import parse_summary_file 
from utils.process_edf_file import process_edf_file

# Filter the specific RuntimeWarning
# warnings.filterwarnings(
#     "ignore", 
#     category=RuntimeWarning, 
#     message="Channel names are not unique, found duplicates for"
# )
# warnings.filterwarnings(
#     "ignore", 
#     category=RuntimeWarning, 
#     message="Scaling factor is not defined in following channels"
# )

########################################################
################# THE MAIN PIPELINE ####################
########################################################

# Define the root directory of your CHB-MIT dataset
SAVED_DATA_FILE = "chb_features_ica.npz"
DATA_DIR = "./PhysioNetData_aws/"

# --- NEW: LOAD-OR-PROCESS LOGIC ---
if os.path.exists(SAVED_DATA_FILE):
    print(f"Loading pre-processed data from {SAVED_DATA_FILE}...")
    data = np.load(SAVED_DATA_FILE)
    X = data['X']
    y = data['y']
    groups = data['groups']
    print("Data loaded successfully.")
else:
    print("No pre-processed data found. Running full processing pipeline...")
    
    master_features_list = []
    master_labels_list = []
    master_patient_groups = [] # For correct train/test split
    
    patient_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "chb*")))
    
    # --- Loop 1: Iterate over each Patient Directory ---
    for patient_dir in patient_dirs:
        if not os.path.isdir(patient_dir):
            continue
            
        patient_id = os.path.basename(patient_dir)
        print(f"\n--- Processing Patient: {patient_id} ---")
        
        # 1. Find and parse the summary file for this patient
        summary_file = glob.glob(os.path.join(patient_dir, "*-summary.txt"))
        if not summary_file:
            print(f"No summary file found for {patient_id}, skipping.")
            continue
        
        seizure_dict = parse_summary_file(summary_file[0])
        
        # 2. Find ALL .edf files for this patient
        edf_files = sorted(glob.glob(os.path.join(patient_dir, "*.edf")))
        
        if not edf_files:
            print(f"No .edf files found for {patient_id}, skipping.")
            continue
        
        # --- Loop 2: Iterate over ALL EDF Files ---
        for edf_path in edf_files:
            filename = os.path.basename(edf_path)
            
            # Get the seizure intervals from our parsed dictionary
            # If the file isn't in the dict, or has no seizures, default to []
            intervals = seizure_dict.get(filename, [])
            
            # 3. Process the file (this is your existing code)
            features, labels = process_edf_file(edf_path, intervals)
            
            # 4. Collect the results (this is your existing code)
            if features is not None and labels is not None:
                master_features_list.append(features)
                master_labels_list.append(labels)
                
                # Keep track of which patient this data belongs to
                patient_group_id = int(patient_id.replace("chb", ""))
                master_patient_groups.append(np.full(len(labels), patient_group_id))
    
    print("\n--- All Files Processed. Combining Data... ---")

    if not master_features_list:
        print("No features were extracted!")
        sys.exit("Stopping script: No data to process.")

    X = np.vstack(master_features_list)
    y = np.concatenate(master_labels_list)
    groups = np.concatenate(master_patient_groups) 
    
    # Save the data for next time
    print(f"Saving processed data to {SAVED_DATA_FILE}...")
    np.savez_compressed(SAVED_DATA_FILE, X=X, y=y, groups=groups)
    print("Data saved successfully.")

print(f"\nTotal features shape (X): {X.shape}")
print(f"Total labels shape (y): {y.shape}")
print(f"Total seizures found (label 1): {np.sum(y)}")

# Check for NaN (Not a Number)
print(f"Is there NaN in data? {np.isnan(X).any()}")

# Check for constant columns (where std = 0)
std_devs = np.std(X, axis=0)
print(f"Number of columns with zero variance: {np.sum(std_devs == 0)}")

########################################################
############ Step 5 & 6: Train, Validate, Save #########
########################################################


# --- CONFIGURATION FROM EXPERIMENT 19 ---
smote_ratio = 0.05
rus_ratio = 0.25
filter_kernel = 5

# Define the pipeline structure (used for both Validation and Final Training)
def get_pipeline(y_train_data=None):
    # 1. Check how many seizures we actually have in this training set
    n_seizures = np.sum(y_train_data == 1)
    
    # 2. Determine Safe Neighbors
    # We need at least 1 neighbor, so k must be < n_seizures
    k_neighbors = min(5, n_seizures - 1)
    
    # 3. Define Steps
    steps = []
    
    steps.append(('selector', VarianceThreshold(threshold=0)))
    # This forces all features to have Mean=0 and Variance=1.
    # This ensures Variance/LineLength don't overpower SMOTE.
    steps.append(('scaler', StandardScaler()))
    
    if k_neighbors > 0:
        steps.append(('smote', SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=42)))
    else:
        steps.append(('ros', RandomOverSampler(sampling_strategy=smote_ratio, random_state=42)))

    steps.append(('rus', RandomUnderSampler(sampling_strategy=rus_ratio, random_state=42)))
    steps.append(('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
    # steps.append(('model', MLPClassifier(
    #     hidden_layer_sizes=(128, 64, 32), # Architecture
    #     activation='relu',                # Activation function
    #     solver='adam',                    # Optimizer (standard)
    #     alpha=0.0001,                     # L2 regularization (prevents overfitting)
    #     learning_rate_init=0.001,         # Learning rate
    #     max_iter=1000,                    # More iterations for convergence
    #     early_stopping=True,              # Stop if no improvement
    #     validation_fraction=0.1,          # Fraction of data for early stopping validation
    #     random_state=42
    # )))

    return Pipeline(steps=steps)


# ======================================================
# PART 1: Validate with THRESHOLD TUNING
# ======================================================
logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, groups)

print(f"\nStarting Cross-Validation on {n_splits} patients...")
print(f"Configuration: SMOTE={smote_ratio}, RUS={rus_ratio}, MedianFilter={filter_kernel}")

# Storage for the "Global" optimization
all_true_labels = []
all_probabilities = [] 
patient_group_map = [] # We need this to re-group data for the median filter later

# --- PASS 1: Collect Probabilities ---
for i, (train_indices, test_indices) in enumerate(logo.split(X, y, groups)):
    test_patient = groups[test_indices][0]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Train Pipeline
    pipeline = get_pipeline(y_train)
    pipeline.fit(X_train, y_train)
    
    # CRITICAL CHANGE: Get Probabilities (Confidence), not just 0/1 predictions
    # Class 1 is the Seizure class
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    all_true_labels.extend(y_test)
    all_probabilities.extend(probs)
    patient_group_map.extend(groups[test_indices]) # Track which patient these probs belong to
    
    print(f"Patient {test_patient} processed.")

# Convert to arrays for calculation
all_true_labels = np.array(all_true_labels)
all_probabilities = np.array(all_probabilities)
patient_group_map = np.array(patient_group_map)

# --- PASS 2: Optimize Threshold Globally ---
print("\n--- Optimizing Threshold on Aggregated Data ---")
precisions, recalls, thresholds = precision_recall_curve(all_true_labels, all_probabilities)

# Calculate F1 for every single possible threshold
with np.errstate(divide='ignore', invalid='ignore'):
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
f1_scores = np.nan_to_num(f1_scores)

# Find the sweet spot
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best Threshold Found: {best_threshold:.4f}")
print(f"Max Theoretical F1:   {best_f1:.4f}")

# Optional: Save the optimization plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.plot(thresholds, f1_scores[:-1], "r-", label="F1 Score")
plt.axvline(best_threshold, color='k', linestyle=':', label=f"Optimal ({best_threshold:.2f})")
plt.title("Threshold Optimization Curve")
plt.legend()
plt.savefig('./threshold_optimization.png')
print("Plot saved to ./Figures/threshold_optimization.png")

# --- PASS 3: Apply Best Threshold & Median Filter ---
print(f"\n--- Re-evaluating with Threshold {best_threshold:.4f} & Filter {filter_kernel} ---")

final_predictions_all = []
final_labels_all = []

# We must apply the Median Filter PER PATIENT (to avoid filtering across patient boundaries)
unique_patients = np.unique(patient_group_map)

for pat in unique_patients:
    # Get indices for this specific patient
    idx = (patient_group_map == pat)
    
    # 1. Retrieve their raw probabilities
    p_probs = all_probabilities[idx]
    
    # 2. Apply the OPTIMIZED Threshold
    p_preds = (p_probs >= best_threshold).astype(int)
    
    # 3. Apply the Median Filter (Smooth out noise)
    p_preds_filtered = medfilt(p_preds, kernel_size=filter_kernel)
    
    final_predictions_all.extend(p_preds_filtered)
    final_labels_all.extend(all_true_labels[idx])

# Final Report
print("\n--- Final Validation Report (Optimized) ---")
print(classification_report(final_labels_all, final_predictions_all, target_names=["Non-Seizure", "Seizure"]))


import joblib
import json

# ======================================================
# PART 2: Final Training & Saving
# ======================================================
print("\n--- Retraining Final Model on ALL Data ---")

# 1. Instantiate a fresh pipeline
# We pass 'y' (all labels) so it knows the total seizure count for SMOTE neighbors
final_pipeline = get_pipeline(y) 

# 2. Fit on the ENTIRE dataset (X, y)
final_pipeline.fit(X, y)

# 3. Save the trained pipeline
model_filename = 'seizure_model_final.pkl'
joblib.dump(final_pipeline, model_filename)

# 4. CRITICAL: Save the Threshold!
# The model file DOES NOT store the threshold. You must save it separately.
config = {
    "best_threshold": float(best_threshold), # The 0.6400 you found
    "filter_kernel": int(filter_kernel),     # The 5 you used
    "smote_ratio": float(smote_ratio),
    "rus_ratio": float(rus_ratio)
}

config_filename = 'model_config.json'
with open(config_filename, 'w') as f:
    json.dump(config, f)

print(f"Model saved to {model_filename}")
print(f"Configuration (Threshold) saved to {config_filename}")