"""
Dataset Structure Setup Script for Human Body Anomaly Detection

This script helps set up the proper directory structure for the skin disease detection model.
It creates the necessary folder structure and provides instructions for downloading datasets.
"""

import os
import shutil
from pathlib import Path
import sys

def create_directory_structure(base_path="Split_smol"):
    """
    Create the necessary directory structure for the skin disease dataset.
    
    Args:
        base_path: The base path for the dataset
    """
    print(f"Creating directory structure at: {base_path}")
    
    # Create base directory
    os.makedirs(base_path, exist_ok=True)
    
    # Create train and validation directories
    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create disease class directories
    disease_classes = [
        "Actinic keratosis",
        "Atopic Dermatitis", 
        "Benign keratosis",
        "Dermatofibroma",
        "Melanocytic nevus",
        "Melanoma",
        "Squamous cell carcinoma",
        "Tinea Ringworm Candidiasis",
        "Vascular lesion"
    ]
    
    for disease in disease_classes:
        os.makedirs(os.path.join(train_dir, disease), exist_ok=True)
        os.makedirs(os.path.join(val_dir, disease), exist_ok=True)
    
    print("Directory structure created successfully.")
    print("\nNext steps:")
    print("1. Download the skin disease dataset from:")
    print("   - ISIC Archive (https://www.isic-archive.com/)")
    print("   - Dermatology Atlas (https://www.dermatologyatlas.net)")
    print("   - DermNet (https://dermnetnz.org/image-library)")
    print("\n2. Place images in their respective disease folders under:")
    print(f"   - {train_dir}/<disease_name>/ (for training images)")
    print(f"   - {val_dir}/<disease_name>/ (for validation images)")
    print("\n3. Make sure to use proper image formats (JPEG, PNG)")

def fix_path_separators(path):
    """
    Ensure correct path separators are used regardless of OS
    """
    return str(Path(path))

def check_existing_structure(base_path="Split_smol"):
    """
    Check if the directory structure already exists
    """
    if os.path.exists(base_path):
        train_dir = os.path.join(base_path, "train")
        val_dir = os.path.join(base_path, "val")
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print(f"Dataset structure already exists at: {base_path}")
            return True
    return False

if __name__ == "__main__":
    base_path = "Split_smol"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    
    # Check if structure exists
    if not check_existing_structure(base_path):
        create_directory_structure(base_path)
    else:
        print("The dataset structure already exists.")
        print(f"If you want to recreate it, please delete the '{base_path}' directory first.") 