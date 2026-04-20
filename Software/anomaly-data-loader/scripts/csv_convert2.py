import pandas as pd
import argparse

def transform_audio_metadata(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # 1. Filter: keep only rows with a Human_validation value
    #df = df[df['Human_validation'].notna()].copy()

    # 2. Fix Filename: reformat ml17_280a_0011_part5.wav → ml17_280a_0011_chunk0005.wav
    pattern = r'(.*)_part(\d+)\.wav'
    extracted = df['Source Audio'].str.extract(pattern)
    df['filename'] = (
        extracted[0] +
        '_chunk' +
        extracted[1].astype(int).apply(lambda x: f"{x:04d}") +
        '.wav'
    )

    # # 3. Map labels: Human_validation TRUE/FALSE → 1/0
    # # pandas reads TRUE/FALSE from CSV as Python booleans
    # label_mapping = {True: 1, False: 0}
    # df['label'] = df['Exploration'].map(label_mapping)
    
    # # 4. Final Cleanup
    # df = df.dropna(subset=['label', 'filename'])
    # df['label'] = df['label'].astype(int)

    # # Move filename and label to the front, keep all other original columns after
    # other_cols = [c for c in df.columns if c not in ('filename', 'label')]
    # df = df[['filename', 'label'] + other_cols]

    df.to_csv(output_csv, index=False)
    print(f"Done! Converted to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform audio metadata CSV for anomaly detection.")
    parser.add_argument("--input-csv", type=str, default="input_file.csv", 
                        help="Path to the input CSV file containing audio metadata.")
    parser.add_argument("--output-csv", type=str, default="converted_labels.csv", 
                        help="Path to save the transformed CSV file with filenames and labels.")
    args = parser.parse_args()
    transform_audio_metadata(args.input_csv, args.output_csv)