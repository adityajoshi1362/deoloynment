from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (e.g., from your HTML on localhost or other domains)

# Load your trained ML pipeline
model = joblib.load('final_pipeline.pkl')  # Make sure this file exists in your Render project root

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('features')  # Expecting a list like [time, current, no2]
        
        if data is None:
            return jsonify({'error': 'No input features provided'}), 400

        input_array = np.array(data).reshape(1, -1)  # Reshape to 2D for sklearn model
        prediction = model.predict(input_array)

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use environment-defined port or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
