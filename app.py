from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import os

# Load dataset
data_path = "data1/housing_prices.csv"
df = pd.read_csv(data_path)

df.dropna(inplace=True)  # Drop missing values to avoid errors

# Define features and target
X = df[["SquareFeet", "NumBedrooms", "NumBathrooms", "GarageSpaces", "YearBuilt", "LocationScore"]]
y = df["Price"]

# Preprocessing: Scaling numerical features
scaler = StandardScaler()

# Build model pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = "model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        input_data = pd.DataFrame({
            "SquareFeet": [float(data["squarefeet"])],
            "NumBedrooms": [int(data["num_bedrooms"])],
            "NumBathrooms": [int(data["num_bathrooms"])],
            "GarageSpaces": [int(data["garage_spaces"])],
            "YearBuilt": [int(data["yearbuilt"])],
            "LocationScore": [float(data["location_score"])]
        })
        
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        prediction = loaded_model.predict(input_data)[0]
        
        return jsonify({"predicted_price": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
