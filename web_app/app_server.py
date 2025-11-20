import os
import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from scipy.signal import medfilt
from utils_web import extract_features_for_inference

# Initialize Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model (Global)
# Note: We must import imblearn so joblib understands the pipeline structure
import imblearn 
MODEL_PATH = "model/seizure_detection_model.joblib"
model = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save file temporarily
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # 1. Process Data
            result, error = extract_features_for_inference(filepath)
            
            # Clean up file
            os.remove(filepath)

            if error:
                return render_template('index.html', error=error)

            X, timestamps = result

            # 2. Run Prediction (Pipeline handles scaling)
            raw_predictions = model.predict(X)

            # 3. Apply "The Logic Fix" (Median Filter)
            # Use same kernel size as training (5)
            final_predictions = medfilt(raw_predictions, kernel_size=5)

            # 4. Format Results
            seizure_events = []
            for i, pred in enumerate(final_predictions):
                if pred == 1:
                    start_t = timestamps[i]
                    end_t = start_t + 5 # 5 second epochs
                    seizure_events.append(f"{start_t:.1f}s - {end_t:.1f}s")

            status = "Seizure Detected!" if len(seizure_events) > 0 else "Normal EEG"
            
            return render_template('index.html', 
                                   status=status, 
                                   events=seizure_events,
                                   filename=file.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)