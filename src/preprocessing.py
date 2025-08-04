import pandas as pd
import os
import logging
from sklearn.impute import KNNImputer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = "data/loaded_data.csv"
OUTPUT_PATH = "data/cleaned_data.csv"

def load_data(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    return pd.read_csv(input_path)

def normalize_text_fields(df):
    """Applies manual mapping to normalize categorical values in HVAC Operation Mode and Activity Level."""
    
    HVAC_Operation_Mode_Mapping = {
        'cooling_active'      : 'Cooling_Active',
        'maintenance_mode'    : 'Maintenance_Mode',
        'heating_active'      : 'Heating_Active',
        'eco_mode'            : 'Eco_Mode',
        'ventilation_only'    : 'Ventilation_Only',
        'off'                 : 'Off',
        'COOLING_ACTIVE'      : 'Cooling_Active',
        'OFF'                 : 'Off',
        'ECO_MODE'            : 'Eco_Mode',
        'Off'                 : 'Off',
        'MAINTENANCE_MODE'    : 'Maintenance_Mode',
        'VENTILATION_ONLY'    : 'Ventilation_Only',
        'HEATING_ACTIVE'      : 'Heating_Active',
        'Heating_Active'      : 'Heating_Active',
        'Ventilation_Only'    : 'Ventilation_Only',
        'Maintenance_Mode'    : 'Maintenance_Mode',
        'Heating_active'      : 'Heating_Active',
        'Maintenance_mode'    : 'Maintenance_Mode',
        'Eco_Mode'            : 'Eco_Mode',
        'Eco_mode'            : 'Eco_Mode',
        'Ventilation_only'    : 'Ventilation_Only',
        'Cooling_active'      : 'Cooling_Active',
        'Cooling_Active'      : 'Cooling_Active',
    }

    Activity_Level_Mapping = {
        'Low Activity'        : 'Low Activity',
        'Moderate Activity'   : 'Moderate Activity',
        'High Activity'       : 'High Activity',
        'ModerateActivity'    : 'Moderate Activity',
        'LowActivity'         : 'Low Activity',
        'Low_Activity'        : 'Low Activity',
    }

    df['HVAC Operation Mode'] = df['HVAC Operation Mode'].map(HVAC_Operation_Mode_Mapping)
    df['Activity Level'] = df['Activity Level'].map(Activity_Level_Mapping)

    logging.info("Normalized text fields: HVAC Operation Mode and Activity Level.")

    return df

def drop_duplicates(df):
    """Drops exact duplicate rows and logs the number removed."""
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        df = df.drop_duplicates()
        logging.info(f"Dropped {num_duplicates} duplicate rows.")
    else:
        logging.info("No duplicate rows found.")
    return df

def drop_session_id(df):
    """Drops the 'Session ID' column if present, since it's not predictive."""
    if 'Session ID' in df.columns:
        df = df.drop(columns=['Session ID'])
        logging.info("Dropped 'Session ID' column.")
    else:
        logging.info("'Session ID' column not found — no action taken.")
    return df

def drop_low_value_features(df):
    """
    Drops features deemed uninformative based on bivariate analysis results.
    Specifically drops: HVAC Operation Mode, Ambient Light Level
    """
    cols_to_drop = ['HVAC Operation Mode', 'Ambient Light Level']
    cols_found = [col for col in cols_to_drop if col in df.columns]

    if cols_found:
        df = df.drop(columns=cols_found)
        logging.info(f"Dropped low-value features: {', '.join(cols_found)}")
    else:
        logging.info("No low-value features found to drop.")
    
    return df

def impute_missing_values(df):
    """
    Handles missing values:
    - Median imputation for CO2_ElectroChemicalSensor
    - KNN imputation for MetalOxideSensor_Unit1-4
    - Row removal for missing CO_GasSensor
    """

    # 1. Drop rows with missing CO_GasSensor
    before = df.shape[0]
    df = df.dropna(subset=['CO_GasSensor'])
    logging.info(f"Dropped {before - df.shape[0]} rows with missing CO_GasSensor values.")

    # 2. Median imputation for CO2_ElectroChemicalSensor
    if 'CO2_ElectroChemicalSensor' in df.columns:
        median_val = df['CO2_ElectroChemicalSensor'].median()
        df.loc[:, 'CO2_ElectroChemicalSensor'] = df['CO2_ElectroChemicalSensor'].fillna(median_val)
        logging.info("Imputed missing values in CO2_ElectroChemicalSensor using median.")

    # 3. KNN imputation for metal oxide sensors
    metal_cols = [
        'MetalOxideSensor_Unit1',
        'MetalOxideSensor_Unit2',
        'MetalOxideSensor_Unit3',
        'MetalOxideSensor_Unit4'
    ]

    if all(col in df.columns for col in metal_cols):
        imputer = KNNImputer(n_neighbors=5)
        df.loc[:, metal_cols] = imputer.fit_transform(df[metal_cols])
        logging.info("Imputed missing values in MetalOxideSensor units using KNN (k=5).")

    return df

def remove_invalid_entries(df):
    """
    Removes rows with invalid sensor readings:
    - Temperature > 280°C
    - Humidity < 0 or > 100
    - CO2_InfraredSensor < 0
    """
    initial_rows = df.shape[0]

    # Remove invalid Temperature readings
    df = df[df['Temperature'] <= 280]

    # Remove invalid Humidity readings
    df = df[(df['Humidity'] >= 0) & (df['Humidity'] <= 100)]

    # Remove invalid CO2_InfraredSensor readings
    df = df[df['CO2_InfraredSensor'] >= 0]

    removed = initial_rows - df.shape[0]
    logging.info(f"Removed {removed} rows with invalid Temperature, Humidity, or CO2_InfraredSensor readings.")

    return df

def remove_outliers(df):
    """
    Removes outliers using both Z-score and IQR filtering.
    Only removes rows that are outliers under both methods.
    """

    z_score_threshold = 4
    iqr_threshold = 2

    numerical_features = [
        'Temperature',
        'Humidity',
        'CO2_InfraredSensor',
        'CO2_ElectroChemicalSensor',
        'MetalOxideSensor_Unit1',
        'MetalOxideSensor_Unit2',
        'MetalOxideSensor_Unit3',
        'MetalOxideSensor_Unit4'
    ]

    for feature in numerical_features:
        initial_count = df.shape[0]

        # IQR method
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_iqr = Q1 - iqr_threshold * IQR
        upper_bound_iqr = Q3 + iqr_threshold * IQR
        outliers_iqr = (df[feature] < lower_bound_iqr) | (df[feature] > upper_bound_iqr)

        # Z-score method
        mean = df[feature].mean()
        std = df[feature].std()
        z_scores = np.abs((df[feature] - mean) / std)
        outliers_zscore = z_scores > z_score_threshold

        # Remove only if both conditions are true
        combined_outliers = outliers_iqr & outliers_zscore
        df = df[~combined_outliers]

        entries_removed = initial_count - df.shape[0]
        logging.info(f"Removed {entries_removed} outliers from {feature}.")

    logging.info(f"Final dataset shape after outlier removal: {df.shape}")
    return df

def ordinal_encode(df):
    """
    Encodes ordered categorical features using ordinal encoding.
    Specifically:
    - Time of Day: Night < Morning < Afternoon < Evening
    - CO_GasSensor: Extreme Low < Low < Medium < High < Extreme High
    """
    if 'Time of Day' in df.columns:
        time_order = ['night', 'morning', 'afternoon', 'evening']
        df['Time of Day'] = pd.Categorical(df['Time of Day'], categories=time_order, ordered=True).codes
        logging.info("Ordinal encoded 'Time of Day'.")

    if 'CO_GasSensor' in df.columns:
        co_order = ['extremely low', 'low', 'medium', 'high', 'extremely high']
        df['CO_GasSensor'] = pd.Categorical(df['CO_GasSensor'], categories=co_order, ordered=True).codes
        logging.info("Ordinal encoded 'CO_GasSensor'.")

    return df

def main():
    logging.info("===== Starting preprocessing pipeline =====")

    df = load_data(INPUT_PATH)
    logging.info(f"Loaded data with shape: {df.shape}")

    df = normalize_text_fields(df)
    df = drop_duplicates(df)
    df = drop_session_id(df)
    df = drop_low_value_features(df)
    df = impute_missing_values(df)
    df = remove_invalid_entries(df)
    df = remove_outliers(df)
    df = ordinal_encode(df)

    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Final preprocessed data saved to: {OUTPUT_PATH}")
    logging.info(f"Final shape: {df.shape}")
    logging.info("===== Preprocessing complete =====")

if __name__ == "__main__":
    main()
