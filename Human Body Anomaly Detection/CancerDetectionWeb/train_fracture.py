# train_fractures.py
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def train_fracture_model(csv_path="D:/Human Body Anomaly Detection/CancerDetectionWeb/fracture_data.csv"):
    """
    Train XGBoost model for fracture detection.
    """
    # Data load
    data = pd.read_csv(csv_path)
    X = data.drop('label', axis=1)
    y = data['label']
    feat_to_keep_fracture = ['mean_intensity', 'variance', 'edge_density']
    
    # Check class balance
    print(f"Fractured samples: {sum(y)}, Non-fractured samples: {len(y) - sum(y)}")
    
    # Model train
    param = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6]
    }
    model = GridSearchCV(
        XGBClassifier(scale_pos_weight=len(y[y==0])/len(y[y==1])),  # Handle imbalance
        param,
        cv=5,
        scoring='f1'
    )
    model.fit(X[feat_to_keep_fracture], y)
    
    # Print best parameters
    print("Best parameters:", model.best_params_)
    
    # Evaluate model
    y_pred = model.predict(X[feat_to_keep_fracture])
    print("Training performance:")
    print(classification_report(y, y_pred))
    
    # Save model
    with open("Fracture_XGBoost", "wb") as f:
        pickle.dump(model, f)
    print("Fracture model saved as 'Fracture_XGBoost'")

if __name__ == '__main__':
    train_fracture_model()