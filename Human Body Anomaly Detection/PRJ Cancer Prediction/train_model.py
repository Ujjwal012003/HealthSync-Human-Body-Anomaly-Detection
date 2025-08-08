import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Data load karo
data = pd.read_csv("D:/Human Body Anomaly Detection/PRJ Cancer Prediction/data.csv")  # Apne data.csv ka path daal do
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})
feat_to_keep = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se']
X_res = X[feat_to_keep]
Y_res = y

# Model train karo
param = {'n_estimators': [100, 500, 1000, 2000]}
model = GridSearchCV(XGBClassifier(), param, cv=10)
model.fit(X_res, Y_res)

# Model save karo
pickle.dump(model, open("XGBoost_new", "wb"))
print("Model saved as 'XGBoost_new'")