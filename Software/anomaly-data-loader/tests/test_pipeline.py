import os
import unittest
import numpy as np
import soundfile as sf
from src.prepare_features import chunk_wav_file
from src.extractor import extract_mel
from src.saver import save_embedding

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_audio_path = 'test.wav'
        self.test_chunk_dir = 'data/audio_chunks'
        self.test_mel_dir = 'data/mels'
        self.test_embedding_dir = 'data/embeddings'
        
        # Create a test audio file
        self.sample_rate = 16000
        self.duration = 10  # seconds
        self.test_audio_data = np.random.rand(self.sample_rate * self.duration)
        sf.write(self.test_audio_path, self.test_audio_data, self.sample_rate)

        # Ensure directories exist
        os.makedirs(self.test_chunk_dir, exist_ok=True)
        os.makedirs(self.test_mel_dir, exist_ok=True)
        os.makedirs(self.test_embedding_dir, exist_ok=True)

    def test_chunk_wav_file(self):
        chunk_wav_file(self.test_audio_path)

        # Check if chunks are created
        chunks = os.listdir(self.test_chunk_dir)
        self.assertGreater(len(chunks), 0, "No audio chunks were created.")

    def test_extract_mel(self):
        mel = extract_mel(self.test_audio_path)
        self.assertEqual(mel.shape[0], 64, "Mel spectrogram should have 64 mel bands.")
        self.assertTrue(np.isfinite(mel).all(), "Mel spectrogram contains non-finite values.")

    def test_save_embedding(self):
        embedding = np.random.rand(128)  # Example embedding
        embedding_path = os.path.join(self.test_embedding_dir, 'test_embedding.npy')
        save_embedding(embedding, embedding_path)

        # Check if embedding is saved correctly
        self.assertTrue(os.path.exists(embedding_path), "Embedding file was not saved.")

    def tearDown(self):
        # Clean up test files and directories
        os.remove(self.test_audio_path)
        for dir_path in [self.test_chunk_dir, self.test_mel_dir, self.test_embedding_dir]:
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
            os.rmdir(dir_path)

if __name__ == '__main__':
    unittest.main()