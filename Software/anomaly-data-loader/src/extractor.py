"""
Perch 2.0 embedding extractor for anomalous sound detection.
Uses ONNX runtime to extract embeddings from audio chunks.
"""
import onnxruntime as ort
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional
from scipy.spatial.distance import cosine

class PerchExtractor:
    """Extract embeddings using Perch 2.0 model."""
    
    def __init__(self, model_path: str = "Software/anomaly-data-loader/src/perch_v2.onnx"):
        """
        Initialize Perch 2.0 extractor.
        
        Args:
            model_path: Path to perch_v2.onnx file
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.session = ort.InferenceSession(str(self.model_path))
        self.sr = 32000  # Perch requires 32kHz
        self.duration = 5.0  # 5 seconds
        self.input_length = int(self.sr * self.duration)  # 160000 samples
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract embedding from an audio file.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            
        Returns:
            Flattened embedding array (shape: (embedding_dim,))
        """
        # Load audio at 32kHz, 5 seconds
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Prepare input tensor (batch_size=1)
        input_tensor = np.zeros((1, self.input_length), dtype=np.float32)
        input_tensor[0, :len(audio)] = audio
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Extract embedding (output[1] contains the embedding)
        embedding = outputs[1].flatten()
        return embedding
    
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine distance between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine distance (0-2 range, lower = more similar)
        """
        return float(cosine(embedding1, embedding2))
    
    def build_reference(self, audio_files: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build reference signature from a list of normal audio files.
        
        Args:
            audio_files: List of paths to normal audio samples
            
        Returns:
            Tuple of (reference_embedding, std_dev)
        """
        embeddings = []
        for audio_path in audio_files:
            try:
                emb = self.extract_embedding(audio_path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Failed to extract embedding from {audio_path}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings extracted")
        
        embeddings_array = np.vstack(embeddings)
        reference = np.mean(embeddings_array, axis=0)
        std_dev = np.std(embeddings_array, axis=0)
        
        return reference, std_dev
    
    def detect_anomaly(
        self, 
        embedding: np.ndarray, 
        reference: np.ndarray, 
        threshold: float = 0.25
    ) -> Tuple[bool, float]:
        """
        Detect anomaly by comparing embedding to reference.
        
        Args:
            embedding: Embedding to check
            reference: Reference embedding from normal audio
            threshold: Distance threshold for anomaly detection
            
        Returns:
            Tuple of (is_anomaly, distance)
        """
        distance = self.compute_distance(embedding, reference)
        is_anomaly = distance > threshold
        return is_anomaly, distance