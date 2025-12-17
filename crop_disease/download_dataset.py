"""
Download and organize the Plant Disease Dataset from Kaggle
This script downloads the dataset and organizes it into train/valid/test folders
"""

import kagglehub
import os
import shutil

# Get the current directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Plant_Disease_Dataset')

def download_dataset():
    """Download the dataset from Kaggle"""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print(f"Downloaded to: {path}")
    return path

def organize_dataset(source_path):
    """Organize the dataset into the required folder structure"""
    
    # Create the main dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # The kaggle dataset structure is typically:
    # new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train
    # new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid
    # We need to find the actual train/valid folders
    
    train_source = None
    valid_source = None
    test_source = None
    
    # Search for train and valid folders in the downloaded path
    for root, dirs, files in os.walk(source_path):
        if 'train' in dirs and train_source is None:
            train_source = os.path.join(root, 'train')
        if 'valid' in dirs and valid_source is None:
            valid_source = os.path.join(root, 'valid')
        if 'test' in dirs and test_source is None:
            test_source = os.path.join(root, 'test')
    
    if train_source is None or valid_source is None:
        # Try alternate structure
        print("Searching for dataset folders...")
        for root, dirs, files in os.walk(source_path):
            print(f"Found directories in {root}: {dirs}")
            if len(dirs) > 10:  # Likely the class folders
                if 'Apple___Apple_scab' in dirs or any('Apple' in d for d in dirs):
                    train_source = root
                    break
    
    print(f"Train source: {train_source}")
    print(f"Valid source: {valid_source}")
    print(f"Test source: {test_source}")
    
    # Define destination paths
    train_dest = os.path.join(DATASET_DIR, 'train')
    valid_dest = os.path.join(DATASET_DIR, 'valid')
    test_dest = os.path.join(DATASET_DIR, 'test', 'test')  # Nested test folder as per original structure
    
    # Copy train folder
    if train_source and os.path.exists(train_source):
        print(f"\nCopying training data...")
        if os.path.exists(train_dest):
            shutil.rmtree(train_dest)
        shutil.copytree(train_source, train_dest)
        print(f"Training data copied to: {train_dest}")
        
        # Count classes and images
        classes = [d for d in os.listdir(train_dest) if os.path.isdir(os.path.join(train_dest, d))]
        total_images = sum(len(os.listdir(os.path.join(train_dest, c))) for c in classes)
        print(f"  - {len(classes)} classes, {total_images} images")
    
    # Copy valid folder
    if valid_source and os.path.exists(valid_source):
        print(f"\nCopying validation data...")
        if os.path.exists(valid_dest):
            shutil.rmtree(valid_dest)
        shutil.copytree(valid_source, valid_dest)
        print(f"Validation data copied to: {valid_dest}")
        
        # Count classes and images
        classes = [d for d in os.listdir(valid_dest) if os.path.isdir(os.path.join(valid_dest, d))]
        total_images = sum(len(os.listdir(os.path.join(valid_dest, c))) for c in classes)
        print(f"  - {len(classes)} classes, {total_images} images")
    
    # Copy or create test folder
    os.makedirs(os.path.join(DATASET_DIR, 'test'), exist_ok=True)
    if test_source and os.path.exists(test_source):
        print(f"\nCopying test data...")
        if os.path.exists(test_dest):
            shutil.rmtree(test_dest)
        shutil.copytree(test_source, test_dest)
        print(f"Test data copied to: {test_dest}")
    else:
        # Create test folder with some sample images from validation
        print(f"\nCreating test folder with sample images from validation...")
        os.makedirs(test_dest, exist_ok=True)
        
        # Copy a few sample images from each class in validation
        if os.path.exists(valid_dest):
            classes = [d for d in os.listdir(valid_dest) if os.path.isdir(os.path.join(valid_dest, d))]
            sample_count = 0
            for cls in classes[:10]:  # Take samples from first 10 classes
                cls_path = os.path.join(valid_dest, cls)
                images = os.listdir(cls_path)[:3]  # Take 3 images per class
                for img in images:
                    src = os.path.join(cls_path, img)
                    # Rename to include class name
                    dst = os.path.join(test_dest, f"{cls}_{img}")
                    shutil.copy2(src, dst)
                    sample_count += 1
            print(f"  - Created {sample_count} test images")

def verify_dataset():
    """Verify the dataset structure"""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    train_path = os.path.join(DATASET_DIR, 'train')
    valid_path = os.path.join(DATASET_DIR, 'valid')
    test_path = os.path.join(DATASET_DIR, 'test')
    
    for name, path in [('Train', train_path), ('Valid', valid_path), ('Test', test_path)]:
        if os.path.exists(path):
            if name == 'Test':
                # Count files in test/test
                test_test = os.path.join(path, 'test')
                if os.path.exists(test_test):
                    files = [f for f in os.listdir(test_test) if os.path.isfile(os.path.join(test_test, f))]
                    print(f"{name}: {len(files)} images")
                else:
                    print(f"{name}: folder exists but test/test not found")
            else:
                classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                total_images = sum(len(os.listdir(os.path.join(path, c))) for c in classes)
                print(f"{name}: {len(classes)} classes, {total_images} images")
        else:
            print(f"{name}: NOT FOUND")
    
    print("\nDataset directory:", DATASET_DIR)

def main():
    print("="*60)
    print("PLANT DISEASE DATASET DOWNLOADER")
    print("="*60)
    print(f"\nTarget directory: {DATASET_DIR}")
    
    # Download the dataset
    source_path = download_dataset()
    
    # Organize into train/valid/test structure
    organize_dataset(source_path)
    
    # Verify the dataset
    verify_dataset()
    
    print("\n✅ Dataset download and organization complete!")
    print("You can now run the training scripts.")

if __name__ == "__main__":
    main()
