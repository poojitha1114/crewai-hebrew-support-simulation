"""
Audio utilities for Hebrew customer service call simulation.
Handles audio file operations, format conversions, and validation.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf
import librosa
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class AudioUtils:
    """Utility class for audio file operations."""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    
    @staticmethod
    def ensure_output_dir(output_dir: str = "output") -> Path:
        """Ensure output directory exists."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        return output_path
    
    @staticmethod
    def generate_audio_filename(speaker: str, turn: int) -> str:
        """Generate timestamped audio filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{speaker}_turn_{turn:03d}_{timestamp}.wav"
    
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """Validate if audio file exists and is readable."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return False
            
            # Try to read audio info
            info = sf.info(file_path)
            logger.info(f"Audio file validated: {file_path} - {info.duration:.2f}s, {info.samplerate}Hz")
            return True
        except Exception as e:
            logger.error(f"Audio validation failed for {file_path}: {e}")
            return False
    
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str, 
                      sample_rate: int = DEFAULT_SAMPLE_RATE,
                      channels: int = DEFAULT_CHANNELS) -> bool:
        """Convert audio file to WAV format with specified parameters."""
        try:
            # Load audio file
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono if needed
            if channels == 1 and audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            audio = audio.set_frame_rate(sample_rate)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            logger.info(f"Audio converted: {input_path} -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"Audio conversion failed: {input_path} -> {output_path}: {e}")
            return False
    
    @staticmethod
    def get_audio_duration(file_path: str) -> Optional[float]:
        """Get audio file duration in seconds."""
        try:
            info = sf.info(file_path)
            return info.duration
        except Exception as e:
            logger.error(f"Failed to get duration for {file_path}: {e}")
            return None
    
    @staticmethod
    def normalize_audio(input_path: str, output_path: str, 
                       target_lufs: float = -23.0) -> bool:
        """Normalize audio to target LUFS level."""
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            
            # Simple RMS normalization (approximation of LUFS)
            rms = librosa.feature.rms(y=y)[0].mean()
            if rms > 0:
                # Normalize to target level
                target_rms = 10 ** (target_lufs / 20)
                y_normalized = y * (target_rms / rms)
                
                # Prevent clipping
                y_normalized = librosa.util.normalize(y_normalized)
                
                # Save normalized audio
                sf.write(output_path, y_normalized, sr)
                logger.info(f"Audio normalized: {input_path} -> {output_path}")
                return True
            else:
                logger.warning(f"Audio file appears to be silent: {input_path}")
                return False
        except Exception as e:
            logger.error(f"Audio normalization failed: {input_path} -> {output_path}: {e}")
            return False
    
    @staticmethod
    def merge_audio_files(file_paths: list, output_path: str, 
                         silence_duration: float = 0.5) -> bool:
        """Merge multiple audio files with silence between them."""
        try:
            combined = AudioSegment.empty()
            silence = AudioSegment.silent(duration=int(silence_duration * 1000))  # Convert to ms
            
            for i, file_path in enumerate(file_paths):
                if not AudioUtils.validate_audio_file(file_path):
                    continue
                
                audio = AudioSegment.from_wav(file_path)
                combined += audio
                
                # Add silence between files (except after the last one)
                if i < len(file_paths) - 1:
                    combined += silence
            
            combined.export(output_path, format="wav")
            logger.info(f"Audio files merged to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Audio merging failed: {e}")
            return False
    
    @staticmethod
    def extract_audio_features(file_path: str) -> dict:
        """Extract basic audio features for quality assessment."""
        try:
            y, sr = librosa.load(file_path, sr=None)
            
            features = {
                'duration': len(y) / sr,
                'sample_rate': sr,
                'rms_energy': float(librosa.feature.rms(y=y)[0].mean()),
                'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(y)[0].mean()),
                'spectral_centroid': float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()),
                'spectral_rolloff': float(librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean())
            }
            
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed for {file_path}: {e}")
            return {}
