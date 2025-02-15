import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Handling missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encoding categorical variables
    if 'Location' in df.columns:
        label_encoder = LabelEncoder()
        df['Location'] = label_encoder.fit_transform(df['Location'])

    # Feature Scaling
    if {'Size', 'Location'}.issubset(df.columns):
        scaler = StandardScaler()
        df[['Size', 'Location']] = scaler.fit_transform(df[['Size', 'Location']])

    df.to_csv(output_path, index=False)
    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    preprocess_data("data1/housing_prices.csv", "data/processed_housing_prices.csv")
