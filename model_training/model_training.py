import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_model(input_path, model_output_path):
    # Load preprocessed data
    df = pd.read_csv(input_path)
    
    # Define features and target variable
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}")
    
    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train_model("data1/processed_housing_prices.csv", "models/housing_price_model.pkl")
