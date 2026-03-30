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

reference_dir = Path("/opt/venv/AnomalySoundDetection/Software/anomaly-data-loader/references")

class PerchExtractor:
    """Extract embeddings using Perch 2.0 model."""
    
    def __init__(self, model_path: str = "/opt/venv/AnomalySoundDetection/Software/Perch2.0/perch_v2.onnx"):
        """
        Initialize Perch 2.0 extractor.
        
        Args:
            model_path: Path to perch_v2.onnx file
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.embedding_dim = self.session.get_outputs()[1].shape[1]  # Assuming output[1] is the embedding
        self.sr = 32000  # Perch requires 32kHz
        self.duration = 5.0  # 5 seconds
        self.input_length = int(self.sr * self.duration)  # 160000 samples
        print(self.session.get_providers())
        
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract embedding from an audio file.

        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)

        Returns:
            Flattened embedding array (shape: (embedding_dim,))
        """
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        return self.extract_embeddings_batch(audio[np.newaxis, :])[0]

    def extract_embeddings_batch(self, audio_arrays: np.ndarray) -> np.ndarray:
        """
        Extract embeddings for a batch of audio arrays (no disk I/O).

        Args:
            audio_arrays: Float32 array of shape (N, input_length) at 32kHz.
                          Shorter arrays are zero-padded automatically.

        Returns:
            Embeddings array of shape (N, embedding_dim)
        """
        N = audio_arrays.shape[0]
        batch = np.zeros((N, self.input_length), dtype=np.float32)
        length = min(audio_arrays.shape[1], self.input_length)
        batch[:, :length] = audio_arrays[:, :length]

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: batch})
        return outputs[1].reshape(N, -1)  # shape: (N, embedding_dim)


    
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
        reference = np.load(reference_dir / "reference_signature.npy") #.mean(embeddings_array, axis=0)
        std_dev = np.load(reference_dir / "reference_std.npy") #.std(embeddings_array, axis=0)
        
        return reference, std_dev
    
    def detect_anomaly(
        self, 
        embedding: np.ndarray, 
        reference: np.ndarray, 
        threshold: float = 0.316
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