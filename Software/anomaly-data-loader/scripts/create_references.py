import sys, argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare_features import main as prepare_main
from prepare_features import build_reference_from_folder as folder2ref

def main():
    
    

