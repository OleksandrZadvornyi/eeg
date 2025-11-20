import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
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

from utils.parse_summary_file import parse_summary_file 
from utils.process_edf_file import process_edf_file

# Filter the specific RuntimeWarning
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message="Channel names are not unique, found duplicates for"
)
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message="Scaling factor is not defined in following channels"
)


########################################################
################# THE MAIN PIPELINE ####################
########################################################

# Define the root directory of your CHB-MIT dataset
SAVED_DATA_FILE = "chb_features.npz"
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

########################################################
############ Step 5 & 6: Train, Validate, Save #########
########################################################


# --- CONFIGURATION FROM EXPERIMENT 19 ---
# This is the "winning" configuration
smote_ratio = 0.025
rus_ratio = 1.0
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
    
    # This forces all features to have Mean=0 and Variance=1.
    # This ensures Variance/LineLength don't overpower SMOTE.
    steps.append(('scaler', StandardScaler()))
    
    if k_neighbors > 0:
        steps.append(('smote', SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=42)))
    else:
        steps.append(('ros', RandomOverSampler(sampling_strategy=smote_ratio, random_state=42)))

    steps.append(('rus', RandomUnderSampler(sampling_strategy=rus_ratio, random_state=42)))
    steps.append(('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))

    return Pipeline(steps=steps)


# ======================================================
# PART 1: Validate on ALL Patients (Cross-Validation)
# ======================================================
# This proves the model works on everyone, not just Patient 1.

logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, groups)
unique_patients = np.unique(groups)

print(f"\nStarting Cross-Validation on {n_splits} patients...")
print(f"Configuration: SMOTE={smote_ratio}, RUS={rus_ratio}, MedianFilter={filter_kernel}")

all_true_labels = []
all_predictions = []

for i, (train_indices, test_indices) in enumerate(logo.split(X, y, groups)):
    test_patient = groups[test_indices][0]
    
    # 1. Split
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # 2. Train Pipeline (SMOTE -> RUS -> Model)
    pipeline = get_pipeline(y_train)
    pipeline.fit(X_train, y_train)
    
    # 3. Predict
    raw_predictions = pipeline.predict(X_test)
    
    # 4. Post-Processing (The "Logic" Fix)
    final_predictions = medfilt(raw_predictions, kernel_size=filter_kernel)
    
    # 5. Store Results
    all_true_labels.extend(y_test)
    all_predictions.extend(final_predictions)
    
    # Optional: Print individual patient progress
    recall = recall_score(y_test, final_predictions, zero_division=0)
    precision = precision_score(y_test, final_predictions, zero_division=0)
    print(f"Patient {test_patient}: Recall={recall:.2f}, Precision={precision:.2f}")

# --- Final Validation Report ---
print("\n--- Final Validation Report (Average across all patients) ---")
print(classification_report(all_true_labels, all_predictions, target_names=["Non-Seizure", "Seizure"]))


# ======================================================
# PART 2: Train & Save "Master Model"
# ======================================================
# Now that we trust the config, we train on EVERYONE to make the best possible model.

print("\nTraining Master Model on ALL data...")

# 1. Create the full pipeline
final_pipeline = get_pipeline(y)

# 2. Train on 100% of the data (X, y)
final_pipeline.fit(X, y)

# 3. Save the model
model_filename = "seizure_detection_model.joblib"
joblib.dump(final_pipeline, model_filename)

print(f"Success! Model saved to {model_filename}")