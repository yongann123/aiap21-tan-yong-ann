import pandas as pd
import os

INPUT_PATH = "data/cleaned_data.csv"
OUTPUT_PATH = "data/featured_data.csv"

def load_data(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    return pd.read_csv(input_path)

def create_features(df):
    # Metal oxide sensor features
    metal_cols = [
        'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2',
        'MetalOxideSensor_Unit3', 'MetalOxideSensor_Unit4'
    ]
    df['metal_oxide_mean'] = df[metal_cols].mean(axis=1)
    df['metal_oxide_std'] = df[metal_cols].std(axis=1)

    # CO2 features
    df['CO2_mean'] = df[['CO2_InfraredSensor', 'CO2_ElectroChemicalSensor']].mean(axis=1)
    df['CO2_diff'] = df['CO2_InfraredSensor'] - df['CO2_ElectroChemicalSensor']

    return df

def main():
    print("Loading preprocessed data...")
    df = load_data(INPUT_PATH)
    print(f"Data shape before feature engineering: {df.shape}")

    df = create_features(df)
    print(f"Data shape after feature engineering: {df.shape}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature-engineered data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
