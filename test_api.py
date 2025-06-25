import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "features": [50,220,80
    ]  # Change values to real input
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Prediction:", response.json())
