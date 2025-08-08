# extract_features.py
import cv2
import numpy as np
import pandas as pd
import os

def extract_features(image_path, label=None):
    """
    Extract mean intensity, variance, and edge density for fracture.
    If label is provided, append to lists for training.
    If label is None, return features for prediction.
    """
    features_list = []
    labels_list = []
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        mean_intensity = np.mean(img)
        variance = np.var(img)
        # Add edge density (fractures mein edges zyada hote hain)
        edges = cv2.Canny(img, 100, 200)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        features = [mean_intensity, variance, edge_density]
        print(f"Extracted features for {image_path}: {features}")
        
        if label is not None:
            features_list.append(features)
            labels_list.append(label)
            return features_list, labels_list
        else:
            return np.array(features)
    else:
        print(f"Failed to load image: {image_path}")
        return None

def process_dataset(fractured_path, non_fractured_path, output_csv):
    """
    Process fracture dataset and save features to CSV.
    """
    features = []
    labels = []
    
    # Fractured images
    print("Processing fractured images...")
    for img_name in os.listdir(fractured_path):
        img_path = os.path.join(fractured_path, img_name)
        feats, lbls = extract_features(img_path, 1)
        if feats:
            features.extend(feats)
            labels.extend(lbls)
    
    # Non-fractured images
    print("Processing non-fractured images...")
    for img_name in os.listdir(non_fractured_path):
        img_path = os.path.join(non_fractured_path, img_name)
        feats, lbls = extract_features(img_path, 0)
        if feats:
            features.extend(feats)
            labels.extend(lbls)
    
    if not features:
        print("No valid features extracted. Check dataset.")
        return
    
    # DataFrame
    df = pd.DataFrame(features, columns=['mean_intensity', 'variance', 'edge_density'])
    df['label'] = labels
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Fracture data saved as '{output_csv}'")
    print(f"Fractured samples: {sum(labels)}, Non-fractured samples: {len(labels) - sum(labels)}")

if __name__ == '__main__':
    fractured_path = "D:/Human Body Anomaly Detection/CancerDetectionWeb/data/FracAtlas/FracAtlas/images/Fractured/"
    non_fractured_path = "D:/Human Body Anomaly Detection/CancerDetectionWeb/data/FracAtlas/FracAtlas/images/Non_fractured/"
    output_csv = "D:/Human Body Anomaly Detection/CancerDetectionWeb/fracture_data.csv"
    process_dataset(fractured_path, non_fractured_path, output_csv)