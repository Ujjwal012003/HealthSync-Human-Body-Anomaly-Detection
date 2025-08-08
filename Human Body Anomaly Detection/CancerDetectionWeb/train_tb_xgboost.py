# train_tb_xgboost.py
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def train_tb_model(csv_path="D:/Human Body Anomaly Detection/CancerDetectionWeb/tb_data.csv"):
    """
    Train XGBoost model for TB detection.
    """
    # Data load
    data = pd.read_csv(csv_path)
    X = data.drop('label', axis=1)
    y = data['label']
    feat_to_keep = ['mean_intensity', 'variance']
    
    # Model train
    param = {'n_estimators': [100, 500]}
    model = GridSearchCV(XGBClassifier(), param, cv=5)
    model.fit(X[feat_to_keep], y)
    
    # Save model
    with open("TB_XGBoost", "wb") as f:
        pickle.dump(model, f)
    print("TB model saved as 'TB_XGBoost'")

if __name__ == '__main__':
    train_tb_model()