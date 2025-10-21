#!/usr/bin/env python3

import shutil
from pathlib import Path
from typing import Optional


def load_ids_from_file(file_path: Path) -> set[str]:
    """Load IDs from a file and return them as a set of lowercase strings."""
    if not file_path:
        return set()
    with open(file_path) as f:
        return {x.lower().strip() for x in f.readlines()}


def split_dataset(
    target_dir: Path,
    output_dir: Path,
    val_split_file: Optional[Path] = None,
    test_split_file: Optional[Path] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split dataset into training, validation and test sets.

    Parameters
    ----------
    target_dir : Path
        Directory containing the .npz structure files
    output_dir : Path
        Directory where to create train, val and test subdirectories
    val_split_file : Optional[Path]
        Path to file containing validation set IDs (one per line)
        If None, no validation set is created
    test_split_file : Optional[Path]
        Path to file containing test set IDs (one per line)
        If None, no test set is created

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Lists of training, validation and test record IDs
    """
    # Setup paths
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load validation and test splits if provided
    val_ids = load_ids_from_file(val_split_file)
    test_ids = load_ids_from_file(test_split_file)

    # Make sure no ID is in both validation and test sets
    common_ids = val_ids & test_ids
    if common_ids:
        print(f"Warning: Found {len(common_ids)} IDs that are in both validation and test sets:")
        print(", ".join(sorted(common_ids)))

    # Split records
    train_records = []
    val_records = []
    test_records = []
    
    # Process all .npz files in the target directory
    for npz_file in target_dir.glob("*.npz"):
        record_id = npz_file.stem  # Get filename without extension
        record_id_lower = record_id.lower()
            
        if record_id_lower in test_ids:
            dst_file = test_dir / npz_file.name
            test_records.append(record_id)
        elif record_id_lower in val_ids:
            dst_file = val_dir / npz_file.name
            val_records.append(record_id)
        else:
            dst_file = train_dir / npz_file.name
            train_records.append(record_id)
            
        shutil.copy2(npz_file, dst_file)

    print(f"Training set: {len(train_records)} structures")
    print(f"Validation set: {len(val_records)} structures")
    print(f"Test set: {len(test_records)} structures")
    
    return train_records, val_records, test_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset into training, validation and test sets")
    parser.add_argument("target_dir", type=str, help="Directory containing the .npz structure files")
    parser.add_argument("output_dir", type=str, help="Output directory for train/val/test splits")
    parser.add_argument("--val-split-file", type=str, help="File containing validation set IDs")
    parser.add_argument("--test-split-file", type=str, help="File containing test set IDs")
    
    args = parser.parse_args()
    
    split_dataset(
        Path(args.target_dir),
        Path(args.output_dir),
        Path(args.val_split_file) if args.val_split_file else None,
        Path(args.test_split_file) if args.test_split_file else None,
    )
