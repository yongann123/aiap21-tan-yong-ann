import sqlite3
import pandas as pd
import os

DATA_PATH = "data/gas_monitoring.db"
OUTPUT_PATH = "data/loaded_data.csv"  # optional

def load_data_from_db(db_path):
    """Loads data from SQLite database into a DataFrame."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM gas_monitoring;"  # Table name is assumed to be 'gas_monitoring'
    
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        conn.close()
        raise RuntimeError(f"Failed to load data from database: {e}")
    
    conn.close()
    return df

def main():
    print("Loading data from database...")
    df = load_data_from_db(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Optionally save to CSV or pickle for use in the next stage
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
