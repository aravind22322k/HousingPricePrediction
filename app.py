from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("data1/housing_prices.csv")

# Define features and target
X = df[["Size", "Location", "Bedrooms", "Bathrooms", "YearBuilt"]]
y = df["Price"]

# Preprocessing: OneHotEncoding for 'Location'
categorical_features = ["Location"]
numeric_features = ["Size", "Bedrooms", "Bathrooms", "YearBuilt"]
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

# Build model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Flask app
app = Flask(_name_)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = pd.DataFrame({
        "Size": [int(data["size"])],
        "Location": [data["location"]],
        "Bedrooms": [int(data["bedrooms"])],
        "Bathrooms": [int(data["bathrooms"])],
        "YearBuilt": [int(data["yearbuilt"])]
    })
    
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    prediction = loaded_model.predict(input_data)[0]
    
    return jsonify({"predicted_price": round(prediction, 2)})

if _name_ == '_main_':
    app.run(debug=True)
