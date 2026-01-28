"""
Convert samples.jsonl to Flex-MoE format (3 modalities: amyloid, mri, demographic)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path


# python convert_to_flexmoe_format.py \
#     --input data/adni_custom/samples.jsonl \
#     --output data/adni_custom/

def load_samples_jsonl(jsonl_path):
    """Load all samples from JSONL file"""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def extract_modality_features(sample, modality_prefix):
    """Extract and flatten all features for a given modality prefix"""
    raw_values = sample['raw_values']
    features = {}
    
    for group_name, group_values in raw_values.items():
        if group_name.startswith(modality_prefix):
            # Add all features from this anatomical group
            features.update(group_values)
    
    return features

def convert_to_flexmoe_format(samples_jsonl_path, output_dir, split_file=None):
    """
    Convert samples.jsonl to Flex-MoE format
    
    Args:
        samples_jsonl_path: Path to samples.jsonl file
        output_dir: Directory to save output CSV files
        split_file: Optional path to split JSON file (e.g., splits_by_ptid_80_10_10.json)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    print(f"Loading samples from {samples_jsonl_path}...")
    samples = load_samples_jsonl(samples_jsonl_path)
    print(f"Loaded {len(samples)} samples")
    
    # Initialize lists
    ptids = []
    amyloid_data = []
    mri_data = []
    demographic_data = []
    labels = []
    
    # Process each sample
    print("\nExtracting features...")
    for sample in samples:
        ptid = sample['PTID']
        ptids.append(ptid)
        
        # Extract features for each modality
        amy_features = extract_modality_features(sample, 'amy_')
        mri_features = extract_modality_features(sample, 'mri_')
        demo_features = sample['raw_values']['demographic']
        
        amyloid_data.append(amy_features)
        mri_data.append(mri_features)
        demographic_data.append(demo_features)
        labels.append(sample['y_true'])
    
    # Convert to DataFrames
    print("\nCreating DataFrames...")
    amyloid_df = pd.DataFrame(amyloid_data, index=ptids)
    mri_df = pd.DataFrame(mri_data, index=ptids)
    demographic_df = pd.DataFrame(demographic_data, index=ptids)
    labels_df = pd.DataFrame({'label': labels}, index=ptids)
    
    # Sort columns for consistency
    amyloid_df = amyloid_df.sort_index(axis=1)
    mri_df = mri_df.sort_index(axis=1)
    demographic_df = demographic_df.sort_index(axis=1)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Number of samples: {len(ptids)}")
    print(f"Amyloid features: {len(amyloid_df.columns)}")
    print(f"MRI features: {len(mri_df.columns)}")
    print(f"Demographic features: {len(demographic_df.columns)}")
    print(f"Label distribution: {labels_df['label'].value_counts().to_dict()}")
    
    # Save to CSV
    print("\nSaving CSV files...")
    amyloid_df.to_csv(output_dir / 'amyloid.csv')
    mri_df.to_csv(output_dir / 'mri.csv')
    demographic_df.to_csv(output_dir / 'demographic.csv')
    labels_df.to_csv(output_dir / 'labels.csv')
    
    print(f"\n✓ Saved to {output_dir}:")
    print(f"  - amyloid.csv ({amyloid_df.shape})")
    print(f"  - mri.csv ({mri_df.shape})")
    print(f"  - demographic.csv ({demographic_df.shape})")
    print(f"  - labels.csv ({labels_df.shape})")
    
    # # If split file provided, copy it
    # if split_file:
    #     import shutil
    #     split_path = Path(split_file)
    #     shutil.copy(split_file, output_dir / split_path.name)
    #     print(f"  - {split_path.name} (copied)")
    
    return amyloid_df, mri_df, demographic_df, labels_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to samples.jsonl')
    parser.add_argument('--output', type=str, default='./data/flexmoe_format',
                        help='Output directory for CSV files')
    parser.add_argument('--split', type=str, default=None,
                        help='Optional: Path to split JSON file')
    
    args = parser.parse_args()
    
    convert_to_flexmoe_format(args.input, args.output, args.split)
