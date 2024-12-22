import requests
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv() # Load environment variables
API_KEY = os.getenv('API_KEY') # API key for OpenWeatherMap

riskExplanation = [["Conditions are favorable for a forest fire if enough vegitation exists in the location."],["Conditions may be suitable for a forest fire if enough vegitation exists in the location."],["Conditions are not favorable for a forest fire."]]

# Define the same model architecture as used during training
class Model(nn.Module):
    def __init__(self, in_features=6, h1=8, h2=9, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Load the trained model
model = Model()  # Initialize the model with the defined architecture
model.load_state_dict(torch.load('./models/forest_fire_model.pth', weights_only=True))  # Load the trained weights into the model
model.eval()  # Set the model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict(): 
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']

    weatherData = getWeatherData(longitude, latitude)
    features = extractFeatures(weatherData)

    # Get predictions
    with torch.no_grad():
        raw_output = model(extractFeatures(weatherData))
        probabilities = torch.softmax(raw_output, dim=0)
        predicted_class = probabilities.argmax().item()

    fireRiskPercentage = probabilities[1].item() * 100
    if (fireRiskPercentage < 40):
        fireRisk = ["Low",2]

    elif (fireRiskPercentage < 65):
        fireRisk = ["Moderate",1]
    else:
        fireRisk = ["High",0]
    
    return jsonify({'risk': fireRisk[0], 'percentage': f"{fireRiskPercentage:.2f}", 'explanation': riskExplanation[fireRisk[1]]})

# URL for weather data API

def getWeatherData(long, lat):
    url = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={long}&units=metric&appid={API_KEY}'

    # Fetch weather data
    response = requests.get(url)
    weatherData = response.json()
    return weatherData

# Extract relevant features
def extractFeatures(weatherData):
    month = int(datetime.datetime.now().month)
    temp = int(weatherData['daily'][0]['temp']['max'])
    rh = int(weatherData['daily'][0]['humidity'])
    wind = int(weatherData['daily'][0]['wind_speed'])
    rain = int(weatherData['daily'][0].get('rain', 0))
    return torch.tensor([month, temp, rh, wind, rain, 5], dtype=torch.float32)

if __name__ == '__main__':
    app.run(debug=True)
