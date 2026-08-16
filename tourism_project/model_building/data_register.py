import os
import pandas as pd

def register_data():
    file_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path)
    print("✅ Dataset successfully registered!")
    print(f"Shape: {df.shape}")
    print("\nColumns present:")
    print(df.columns.tolist())

if __name__ == "__main__":
    register_data()
