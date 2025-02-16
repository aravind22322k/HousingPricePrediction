from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(_name_)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    size = float(request.form["size"])
    location = request.form["location"]

    location_encoded = 1 if location == "Urban" else (2 if location == "Suburban" else 3)
    input_data = pd.DataFrame([[size, location_encoded]], columns=["Size", "Location"])

    prediction = model.predict(input_data)[0]

    return f"<h3>Predicted House Price: ${prediction:,.2f}</h3>"

if _name_ == "_main_":
    app.run(debug=True)
