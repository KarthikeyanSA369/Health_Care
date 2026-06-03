import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

def train_diabetes():
    print("Training Diabetes Model...")
    df = pd.read_csv("dataset/diabetes.csv")
    
    # Cleaning
    cols_clean = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[cols_clean] = df[cols_clean].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    
    # Features matching HTML form order
    features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']
    
    X = df[features]
    y = df['Outcome']
    
    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # GridSearch
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    print(f"Diabetes Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    
    pickle.dump(model, open("model/diabetes_model.pkl", "wb"))
    pickle.dump(scaler, open("model/diabetes_scaler.pkl", "wb"))
    print("Diabetes model and scaler saved.\n")

def train_heart():
    print("Training Heart Model...")
    df = pd.read_csv("dataset/heart.csv")
    df.fillna(df.median(), inplace=True)
    
    # Features matching HTML form order
    features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    
    X = df[features]
    y = df["target"]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Heart Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    
    pickle.dump(model, open("model/heart_model.pkl", "wb"))
    pickle.dump(scaler, open("model/heart_scaler.pkl", "wb"))
    print("Heart model and scaler saved.\n")

def train_obesity():
    print("Training Obesity Model...")
    df = pd.read_csv("dataset/obesity.csv")
    
    # Cleaning / Encoding Target
    le = LabelEncoder()
    df["NObeyesdad"] = le.fit_transform(df["NObeyesdad"])
    
    # Define features matching HTML form (Intersection with dataset)
    # Form: Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE
    # Data: Gender, Age, ..., FCVC, NCP, CH2O, FAF, TUE, ...
    # Overlap: Age, FCVC, NCP, CH2O, FAF, TUE
    # Missing from Data: Height, Weight (Cannot use these)
    
    features = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    
    # Verify these columns exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"Warning: Obesity columns missing: {missing}")
        return

    X = df[features]
    y = df["NObeyesdad"]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Obesity Accuracy (Simplified Features): {accuracy_score(y_test, model.predict(X_test)):.4f}")
    
    pickle.dump(model, open("model/obesity_model.pkl", "wb"))
    pickle.dump(scaler, open("model/obesity_scaler.pkl", "wb"))
    print("Obesity model and scaler saved.\n")

if __name__ == "__main__":
    train_diabetes()
    train_heart()
    train_obesity()
