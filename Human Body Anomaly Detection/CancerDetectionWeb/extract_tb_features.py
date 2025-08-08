# extract_tb_features.py
import cv2
import numpy as np
import pandas as pd
import os

def extract_features(image_path, label=None):
    """
    Extract mean intensity and variance for TB.
    If label is provided, append to lists for training.
    If label is None, return features for prediction.
    """
    features_list = []
    labels_list = []
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        mean_intensity = np.mean(img)
        variance = np.var(img)
        if label is not None:
            features_list.append([mean_intensity, variance])
            labels_list.append(label)
            return features_list, labels_list
        else:
            return np.array([mean_intensity, variance])
    else:
        return None

def process_dataset(normal_path, tb_path, output_csv):
    """
    Process TB dataset and save features to CSV.
    """
    features = []
    labels = []
    
    # Normal images
    for img_name in os.listdir(normal_path):
        img_path = os.path.join(normal_path, img_name)
        feats, lbls = extract_features(img_path, 0)
        if feats:
            features.extend(feats)
            labels.extend(lbls)
    
    # TB images
    for img_name in os.listdir(tb_path):
        img_path = os.path.join(tb_path, img_name)
        feats, lbls = extract_features(img_path, 1)
        if feats:
            features.extend(feats)
            labels.extend(lbls)
    
    # DataFrame
    df = pd.DataFrame(features, columns=['mean_intensity', 'variance'])
    df['label'] = labels
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"TB data saved as '{output_csv}'")

if __name__ == '__main__':
    normal_path = "D:/Human Body Anomaly Detection/CancerDetectionWeb/data/tb/Normal/"
    tb_path = "D:/Human Body Anomaly Detection/CancerDetectionWeb/data/tb/Tuberculosis/"
    output_csv = "D:/Human Body Anomaly Detection/CancerDetectionWeb/tb_data.csv"
    process_dataset(normal_path, tb_path, output_csv)