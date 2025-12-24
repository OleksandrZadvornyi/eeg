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

# Global configuration
SAVED_DATA_FILE = "chb_features_ica.npz"
DATA_DIR = "./CHB-MIT_Scalp_EEG_Database/"

# --- Data Loading / Preprocessing ---
if os.path.exists(SAVED_DATA_FILE):
    print(f"Loading pre-processed data from {SAVED_DATA_FILE}...")
    data = np.load(SAVED_DATA_FILE)
    X, y, groups = data['X'], data['y'], data['groups']
    print("Data loaded successfully.")
else:
    print("No pre-processed data found. Extracting features...")
    
    master_features_list = []
    master_labels_list = []
    master_patient_groups = [] 
    
    patient_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "chb*")))
    
    for patient_dir in patient_dirs:
        if not os.path.isdir(patient_dir):
            continue
            
        patient_id = os.path.basename(patient_dir)
        print(f"\nProcessing Patient: {patient_id}")
        
        summary_file = glob.glob(os.path.join(patient_dir, "*-summary.txt"))
        if not summary_file:
            continue
        
        seizure_dict = parse_summary_file(summary_file[0])
        edf_files = sorted(glob.glob(os.path.join(patient_dir, "*.edf")))
        
        if not edf_files:
            continue
        
        for edf_path in edf_files:
            filename = os.path.basename(edf_path)
            intervals = seizure_dict.get(filename, [])
            
            features, labels = process_edf_file(edf_path, intervals)
            
            if features is not None and labels is not None:
                master_features_list.append(features)
                master_labels_list.append(labels)
                
                # Assign patient ID for GroupKFold/LOGO
                patient_group_id = int(patient_id.replace("chb", ""))
                master_patient_groups.append(np.full(len(labels), patient_group_id))
    
    if not master_features_list:
        sys.exit("Extraction failed: No data found.")

    X = np.vstack(master_features_list)
    y = np.concatenate(master_labels_list)
    groups = np.concatenate(master_patient_groups) 
    
    np.savez_compressed(SAVED_DATA_FILE, X=X, y=y, groups=groups)

# Data quality checks
print(f"X shape: {X.shape} | y shape: {y.shape} | Seizure count: {np.sum(y)}")
print(f"Missing values: {np.isnan(X).any()}")
std_devs = np.std(X, axis=0)
print(f"Zero variance columns: {np.sum(std_devs == 0)}")

# --- Training Configuration ---
smote_ratio = 0.05
rus_ratio = 0.25
filter_kernel = 5

def get_pipeline(y_train_data=None):
    """Constructs the resampling and classification pipeline."""
    n_seizures = np.sum(y_train_data == 1)
    k_neighbors = min(5, n_seizures - 1)
    
    steps = []
    steps.append(('selector', VarianceThreshold(threshold=0)))
    steps.append(('scaler', StandardScaler()))
    
    # Select oversampling method based on minority class size
    if k_neighbors > 0:
        steps.append(('smote', SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=42)))
    else:
        steps.append(('ros', RandomOverSampler(sampling_strategy=smote_ratio, random_state=42)))

    steps.append(('rus', RandomUnderSampler(sampling_strategy=rus_ratio, random_state=42)))
    steps.append(('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))

    return Pipeline(steps=steps)

# --- Cross-Validation & Threshold Tuning ---
logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, groups)

all_true_labels = []
all_probabilities = [] 
patient_group_map = [] 

print(f"Executing LOGO Cross-Validation ({n_splits} folds)...")

for i, (train_indices, test_indices) in enumerate(logo.split(X, y, groups)):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    pipeline = get_pipeline(y_train)
    pipeline.fit(X_train, y_train)
    
    # Store probabilities for precision-recall optimization
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    all_true_labels.extend(y_test)
    all_probabilities.extend(probs)
    patient_group_map.extend(groups[test_indices])

all_true_labels = np.array(all_true_labels)
all_probabilities = np.array(all_probabilities)
patient_group_map = np.array(patient_group_map)

# Optimize threshold for F1-Score
precisions, recalls, thresholds = precision_recall_curve(all_true_labels, all_probabilities)
with np.errstate(divide='ignore', invalid='ignore'):
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
f1_scores = np.nan_to_num(f1_scores)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Visualize optimization
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.plot(thresholds, f1_scores[:-1], "r-", label="F1 Score")
plt.axvline(best_threshold, color='k', linestyle=':', label=f"Optimal ({best_threshold:.2f})")
plt.title("Threshold Optimization")
plt.legend()
plt.savefig('./threshold_optimization.png')

# --- Evaluation with Temporal Filtering ---
final_predictions_all = []
final_labels_all = []
unique_patients = np.unique(patient_group_map)

for pat in unique_patients:
    idx = (patient_group_map == pat)
    p_probs = all_probabilities[idx]
    
    # Apply optimal threshold and median filter per patient session
    p_preds = (p_probs >= best_threshold).astype(int)
    p_preds_filtered = medfilt(p_preds, kernel_size=filter_kernel)
    
    final_predictions_all.extend(p_preds_filtered)
    final_labels_all.extend(all_true_labels[idx])

print("\nFinal Metrics (Optimized):")
print(classification_report(final_labels_all, final_predictions_all, target_names=["Non-Seizure", "Seizure"]))

# --- Final Export ---
print("\nRetraining on full dataset...")
final_pipeline = get_pipeline(y) 
final_pipeline.fit(X, y)

joblib.dump(final_pipeline, 'seizure_model_final.pkl')

# Save decision parameters
config = {
    "best_threshold": float(best_threshold),
    "filter_kernel": int(filter_kernel),
    "smote_ratio": float(smote_ratio),
    "rus_ratio": float(rus_ratio)
}

with open('model_config.json', 'w') as f:
    json.dump(config, f)