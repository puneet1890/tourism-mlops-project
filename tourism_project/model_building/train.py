import os
import joblib
import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score

def train_model():
    # 1. Load split dataset artifacts
    train_df = pd.read_csv("tourism_project/data/train.csv")
    test_df = pd.read_csv("tourism_project/data/test.csv")
    
    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]
    
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    # 2. Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    
    # 3. Track experiment parameters and metrics with MLflow
    mlflow.set_experiment("Wellness_Tourism_Experiment")
    
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # 4. Export best model artifact for deployment
        os.makedirs("tourism_project/deployment", exist_ok=True)
        model_path = "tourism_project/deployment/best_model.pkl"
        joblib.dump(pipeline, model_path)
        
        print(f"✅ Training Complete. F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"✅ Saved model artifact to: {model_path}")

if __name__ == "__main__":
    train_model()
