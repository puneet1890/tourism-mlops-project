import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # 1. Load dataset from repository data folder
    df = pd.read_csv("tourism_project/data/tourism.csv")
    
    # 2. Remove unnecessary identifier column
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    # Standardize categorical typo if present
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
        
    # Handle missing values
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # 3. Separate features and target
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]
    
    # 4. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Save split files locally as CSVs
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv("tourism_project/data/train.csv", index=False)
    test_data.to_csv("tourism_project/data/test.csv", index=False)
    
    print("✅ Successfully generated and saved train.csv and test.csv in tourism_project/data/")

if __name__ == "__main__":
    prepare_data()
