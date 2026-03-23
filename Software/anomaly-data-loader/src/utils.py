from pathlib import Path
import os

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_audio_chunk(chunk, output_path, sample_rate=16000):
    """Save a 5-second audio chunk to the specified path."""
    chunk.export(output_path, format="wav")

def save_mel_spectrogram(mel, output_path):
    """Save the mel spectrogram as a numpy file."""
    np.save(output_path, mel)

def save_embedding(embedding, output_path):
    """Save the embedding as a numpy file."""
    np.save(output_path, embedding)

def load_audio_file(file_path):
    """Load an audio file and return the audio data and sample rate."""
    return librosa.load(file_path, sr=None)

def get_file_paths(directory, extension):
    """Get all file paths in a directory with the specified extension."""
    return list(Path(directory).rglob(f'*.{extension}'))