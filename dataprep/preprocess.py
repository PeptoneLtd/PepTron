import os
import shutil
import random

def reorganize_files(source_dir, dest_dir, train_ratio=0.7):
    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    valid_dir = os.path.join(dest_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Collect all .cif files recursively
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.cif'):
                all_files.append(os.path.join(root, file))

    # Randomly shuffle files
    random.shuffle(all_files)

    # Calculate split point
    split_idx = int(len(all_files) * train_ratio)

    # Split files into train and valid sets
    train_files = all_files[:split_idx]
    valid_files = all_files[split_idx:]

    # Copy files to respective directories
    for file in train_files:
        filename = os.path.basename(file)
        shutil.copy2(file, os.path.join(train_dir, filename))

    for file in valid_files:
        filename = os.path.basename(file)
        shutil.copy2(file, os.path.join(valid_dir, filename))

    print(f"Total files: {len(all_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")

# Usage
if __name__ == "__main__":
    source_directory = "/mnt/data/datasets/pdb_mmcif"
    destination_directory = "/mnt/data/datasets/pdb_mmcif_full"
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    reorganize_files(source_directory, destination_directory)