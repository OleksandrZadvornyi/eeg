# In parse_summary_file.py
import re

def parse_summary_file(summary_path):
    """
    Parses a chbXX-summary.txt file.
    
    Returns a dictionary where:
    - keys are filenames (e.g., "chb01_03.edf")
    - values are lists of (start, end) seizure tuples
    """
    seizure_dict = {}
    current_filename = ""
    start_sec = None  # Temporary variable to hold the start time

    try:
        with open(summary_path, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith("File Name:"):
                    current_filename = line.split(":")[-1].strip()
                    seizure_dict[current_filename] = [] # Initialize with no seizures
                    start_sec = None # Reset on new file
                
                # Use regex to find "Seizure X Start Time:" or "Seizure Start Time:"
                elif re.match(r"^Seizure \d* ?Start Time:", line):
                    start_sec = int(line.split(":")[-1].strip().split(" ")[0])
                
                # Use regex to find "Seizure X End Time:" or "Seizure End Time:"
                elif re.match(r"^Seizure \d* ?End Time:", line):
                    # Make sure we have a start time to match
                    if start_sec is not None:
                        end_sec = int(line.split(":")[-1].strip().split(" ")[0])
                        seizure_dict[current_filename].append((start_sec, end_sec))
                        start_sec = None # Reset for the next seizure
    
    except Exception as e:
        print(f"Error parsing summary file {summary_path}: {e}")
    
    return seizure_dict