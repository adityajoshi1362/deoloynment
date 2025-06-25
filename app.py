from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your pipeline
model = joblib.load('final_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('features')  # JSON must have "features"
        if data is None:
            return jsonify({'error': 'No input features provided'}), 400

        input_array = np.array(data).reshape(1, -1)
        prediction = model.predict(input_array)

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
