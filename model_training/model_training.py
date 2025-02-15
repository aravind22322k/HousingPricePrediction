import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)

    # Splitting data
    X = df[['Size', 'Location']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained! MAE: {mae:.2f}")

    # Save model
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_model("data/processed_housing_prices.csv", "model.pkl")
 
