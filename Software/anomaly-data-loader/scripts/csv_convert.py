import pandas as pd
import argparse

def transform_audio_metadata(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    # 1. Filter: Keep rows where Human_validation exists OR Status is 'Void'
    # We use a boolean OR (|) condition here
    mask = (df['Human_validation'].notna()) | (df['Status'] == 'Void')
    df = df[mask].copy()
    
    # 2. Fix Filename: Extract prefix and part number, then format as 4-digit chunk
    # This regex captures the base name and the number after '_part'
    pattern = r'(.*)_part(\d+)\.wav'
    extracted = df['Source Audio'].str.extract(pattern)
    
    # extracted[0] is the base (ml17_280a_0099)
    # extracted[1] is the number (41)
    df['filename'] = (
        extracted[0] + 
        '_chunk' + 
        extracted[1].astype(int).apply(lambda x: f"{x:04d}") + 
        '.wav'
    )
    
   # 3. Map Labels: Define how text categories translate to numbers
    # Mapping 'Void' to 0 here; change to 1 if you consider it an anomaly
    label_mapping = {
        'Normal': 0,
        'Anomaly': 1,
        'Void': 0  
    }
    
    # We check both columns for the label (Human_validation takes priority)
    df['label'] = df['Human_validation'].fillna(df['Status']).map(label_mapping)
    
    # 4. Final Cleanup
    df = df.dropna(subset=['label', 'filename'])
    final_df = df[['filename', 'label']].copy()
    final_df['label'] = final_df['label'].astype(int)
    
    final_df.to_csv(output_csv, index=False)
    print(f"Done! Converted to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform audio metadata CSV for anomaly detection.")
    parser.add_argument("--input-csv", type=str, default="input_file.csv", 
                        help="Path to the input CSV file containing audio metadata.")
    parser.add_argument("--output-csv", type=str, default="converted_labels.csv", 
                        help="Path to save the transformed CSV file with filenames and labels.")
    args = parser.parse_args()
    transform_audio_metadata(args.input_csv, args.output_csv)