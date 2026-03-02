"""
Build reference embeddings from normal/clean audio samples.
Used for anomaly detection baseline.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from extractor import PerchExtractor

def build_reference(audio_folder, output_dir="data"):
    """
    Build reference embedding from a folder of normal audio files.
    
    Args:
        audio_folder: Path to folder containing normal 5s WAV files
        output_dir: Directory to save reference files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ref_emb_path = output_path / "reference_embedding.npy"
    ref_std_path = output_path / "reference_std.npy"
    
    print(f"Building reference from: {audio_folder}")
    
    # Initialize Perch extractor
    try:
        perch = PerchExtractor()
    except Exception as e:
        print(f"Error loading Perch model: {e}")
        return
    
    # Collect audio files
    audio_files = list(Path(audio_folder).glob("*.wav"))
    if not audio_files:
        print(f"No WAV files found in {audio_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Extract embeddings
    embeddings = []
    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        try:
            emb = perch.extract_embedding(str(audio_path))
            embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            continue
    
    if not embeddings:
        print("No embeddings extracted")
        return
    
    # Compute reference
    embeddings_array = np.vstack(embeddings)
    reference = np.mean(embeddings_array, axis=0)
    std_dev = np.std(embeddings_array, axis=0)
    
    # Save
    np.save(ref_emb_path, reference)
    np.save(ref_std_path, std_dev)
    
    print(f"\nReference embedding saved to: {ref_emb_path}")
    print(f"   Embedding dimension: {reference.shape[0]}")
    print(f"   Number of samples: {len(embeddings)}")
    print(f"   Mean stability (std): {np.mean(std_dev):.4f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ref_builder.py <audio_folder> [output_dir]")
        print("Example: python ref_builder.py data/normal_audio data")
        sys.exit(1)
    
    audio_folder = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    build_reference(audio_folder, output_dir)
