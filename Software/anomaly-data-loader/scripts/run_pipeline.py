"""
Complete pipeline for anomalous sound detection data preparation.
Process raw audio files, extract features (mel, embeddings), and generate metadata.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare_features import main as prepare_main
from ref_builder import build_reference

def main():
    """Run complete data preparation pipeline."""
    print("=" * 60)
    print("Anomaly Sound Detection - Data Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Process and chunk raw audio
    print("\n[Step 1/2] Processing raw audio files...")
    print("-" * 60)
    try:
        prepare_main()
    except Exception as e:
        print(f"Error during feature preparation: {e}")
        return False
    
    # Step 2: Optional - Build reference from normal audio
    print("\n[Step 2/2] Building reference embedding (optional)...")
    print("-" * 60)
    normal_audio_path = Path("data/audio_chunks/normal")
    if normal_audio_path.exists() and list(normal_audio_path.glob("*.wav")):
        try:
            print(f"Found normal audio samples in {normal_audio_path}")
            build_reference(str(normal_audio_path), "data")
            print("Reference embedding built successfully")
        except Exception as e:
            print(f"Note: Could not build reference: {e}")
            print("   You can build it later using: python -m src.ref_builder")
    else:
        print(f"No normal audio found in {normal_audio_path}")
        print("   (Optional) To use anomaly detection, prepare normal audio in:")
        print(f"   {normal_audio_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print("\nOutput structure:")
    print("  data/")
    print("    ├── audio_chunks/       (5s WAV files)")
    print("    ├── mels/               (mel-spectrograms)")
    print("    ├── embeddings/         (Perch 2.0 embeddings)")
    print("    ├── metadata.csv        (main metadata file)")
    print("    ├── reference_embedding.npy  (optional)")
    print("    └── reference_std.npy        (optional)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)