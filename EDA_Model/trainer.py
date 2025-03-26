import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_model():
    print("Loading dataset...")
    try:
        # Load dataset
        df = pd.read_csv("EDA_Model/Model_test_data/DOE.csv", encoding="latin1")
    except FileNotFoundError:
        print("Dataset file not found. Please check the file path.")
        return
    
    print("Preprocessing data...")
    # Data Preprocessing
    if "EC(µS/cm)" in df.columns:
        df["EC(µS/cm)"] = pd.to_numeric(df["EC(µS/cm)"], errors="coerce")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    columns_to_drop = ["Sample Location", "Lab code", "Date"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")

    # Define features and target variable
    X = df.drop(columns=["DO(mg/l)"])
    y = df["DO(mg/l)"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scalar.joblib')
    
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_model()

