import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
import os  
import glob
import warnings
import sys

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
############ Step 5 & 6: Train and Evaluate ############
########################################################

# --- The RIGHT Way to Split: By Patient ---
logo = LeaveOneGroupOut()

# Check if we have more than one patient, otherwise split
if len(np.unique(groups)) > 1:
    train_indices, test_indices = next(logo.split(X, y, groups))
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    test_patient = groups[test_indices][0]
    
    print(f"\nSplitting data. Testing on Patient {test_patient}.")
else:
    # Fallback for testing with only one patient's data
    print("\nWarning: Only one patient group found. Splitting randomly (not for valid results).")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

print(f"Training samples: {len(X_train)} (Seizures: {np.sum(y_train)})")
print(f"Testing samples: {len(X_test)}")


# We'll create a 1:1 ratio (or as close as possible) of seizure to non-seizure
# Other good ratios to try are 0.5 (1 seizure per 2 non-seizure) or 0.25
rus = RandomUnderSampler(sampling_strategy=0.17, random_state=42)
print("Resampling the training data...")
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print(f"Resampled training samples: {len(X_train_resampled)} (Seizures: {np.sum(y_train_resampled)})")
# --- END OF ADDITION ---

# --- Train ---
print("\nTraining the model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1 # Use all available CPU cores
)
model.fit(X_train_resampled, y_train_resampled)

# --- Evaluate ---
print("Making predictions on the test data...")
predictions = model.predict(X_test)

print("\n--- Classification Report ---")
report = classification_report(y_test, predictions, target_names=["Non-Seizure (0)", "Seizure (1)"], zero_division=0)
print(report)

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, predictions)
print(cm)

# You can use this to create a heatmap visualization
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Non-Seizure', 'Predicted Seizure'], 
            yticklabels=['Actual Non-Seizure', 'Actual Seizure'])
plt.title('Confusion Matrix')
plt.show()