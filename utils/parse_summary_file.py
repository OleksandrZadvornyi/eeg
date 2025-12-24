import re

def parse_summary_file(summary_path):
    """
    Parses CHB-MIT summary text files to extract seizure timestamps.
    
    Returns:
        dict: {filename: [(start_1, end_1), (start_2, end_2), ...]}
    """
    seizure_dict = {}
    current_filename = ""
    start_sec = None

    try:
        with open(summary_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Identify entry for a new EDF file
                if line.startswith("File Name:"):
                    current_filename = line.split(":")[-1].strip()
                    seizure_dict[current_filename] = []
                    start_sec = None 
                
                # Extract seizure start timestamp
                elif re.match(r"^Seizure \d* ?Start Time:", line):
                    start_sec = int(line.split(":")[-1].strip().split(" ")[0])
                
                # Extract seizure end timestamp and pair with start time
                elif re.match(r"^Seizure \d* ?End Time:", line):
                    if start_sec is not None:
                        end_sec = int(line.split(":")[-1].strip().split(" ")[0])
                        seizure_dict[current_filename].append((start_sec, end_sec))
                        start_sec = None 
    
    except Exception as e:
        print(f"Error parsing summary file {summary_path}: {e}")
    
    return seizure_dict