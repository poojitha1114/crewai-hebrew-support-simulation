"""
Simple Audio Merger - Alternative approach using wave module for audio concatenation.
"""

import os
import json
import wave
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleAudioMerger:
    """Simple audio merger using Python's built-in wave module."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the audio merger."""
        self.output_dir = Path(output_dir)
    
    def find_conversation_files(self, conversation_id: str) -> Dict[int, Dict[str, str]]:
        """Find all audio files for a specific conversation."""
        audio_files = {}
        
        # Extract timestamp from conversation_id
        if conversation_id.startswith("conv_"):
            timestamp = conversation_id[5:]
        else:
            timestamp = conversation_id
        
        # Find all audio files matching this conversation
        for audio_file in self.output_dir.glob("*.wav"):
            filename = audio_file.name
            
            # Skip merged conversation files
            if filename.startswith("conversation_complete_"):
                continue
            
            if timestamp in filename:
                parts = filename.split('_')
                if len(parts) >= 4:
                    speaker = parts[0]  # client or csr
                    turn_str = parts[2]  # turn number
                    
                    if turn_str.isdigit():
                        turn_num = int(turn_str)
                        
                        if turn_num not in audio_files:
                            audio_files[turn_num] = {}
                        
                        audio_files[turn_num][speaker] = str(audio_file)
        
        return audio_files
    
    def find_latest_conversation(self) -> Optional[str]:
        """Find the most recent conversation ID from transcript files."""
        transcript_files = list(self.output_dir.glob("transcript_*.json"))
        
        if not transcript_files:
            return None
        
        latest_transcript = max(transcript_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_transcript, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('metadata', {}).get('conversation_id')
        except Exception as e:
            logger.error(f"Failed to read transcript {latest_transcript}: {e}")
            return None
    
    def is_valid_wav_file(self, file_path: str) -> bool:
        """Check if a WAV file is valid and readable."""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                # Try to read basic properties
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                return frames > 0 and sample_rate > 0 and channels > 0
        except Exception:
            return False
    
    def merge_wav_files(self, file_paths: List[str], output_path: str, silence_duration: float = 0.5) -> bool:
        """
        Merge multiple WAV files into one using the wave module.
        
        Args:
            file_paths: List of WAV file paths to merge
            output_path: Output file path
            silence_duration: Seconds of silence between files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter out invalid files
            valid_files = [f for f in file_paths if os.path.exists(f) and self.is_valid_wav_file(f)]
            
            if not valid_files:
                logger.warning("No valid audio files found to merge")
                return False
            
            # Get audio parameters from the first valid file
            with wave.open(valid_files[0], 'rb') as first_wav:
                params = first_wav.getparams()
                sample_rate = params.framerate
                channels = params.nchannels
                sample_width = params.sampwidth
            
            # Calculate silence frames
            silence_frames = int(silence_duration * sample_rate)
            silence_data = b'\x00' * (silence_frames * channels * sample_width)
            
            # Create output file
            with wave.open(output_path, 'wb') as output_wav:
                output_wav.setparams(params)
                
                for i, file_path in enumerate(valid_files):
                    try:
                        with wave.open(file_path, 'rb') as input_wav:
                            # Copy audio data
                            data = input_wav.readframes(input_wav.getnframes())
                            output_wav.writeframes(data)
                            
                            # Add silence between files (except after the last file)
                            if i < len(valid_files) - 1:
                                output_wav.writeframes(silence_data)
                            
                            logger.info(f"Added audio file: {Path(file_path).name}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {e}")
                        continue
            
            logger.info(f"Successfully merged {len(valid_files)} audio files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge audio files: {e}")
            return False
    
    def merge_conversation_audio(self, conversation_id: str = None, silence_duration: float = 0.5) -> Optional[str]:
        """
        Merge all audio files from a conversation into one complete audio file.
        
        Args:
            conversation_id: Specific conversation to merge (if None, uses latest)
            silence_duration: Seconds of silence between turns
            
        Returns:
            Path to the merged audio file or None if failed
        """
        if conversation_id is None:
            conversation_id = self.find_latest_conversation()
            if conversation_id is None:
                logger.error("No conversations found")
                return None
        
        logger.info(f"Merging audio for conversation: {conversation_id}")
        
        # Find all audio files for this conversation
        audio_files = self.find_conversation_files(conversation_id)
        
        if not audio_files:
            logger.error(f"No audio files found for conversation {conversation_id}")
            return None
        
        # Create ordered list of audio files (client first, then CSR for each turn)
        ordered_files = []
        for turn_num in sorted(audio_files.keys()):
            turn_files = audio_files[turn_num]
            
            # Add client audio first
            if 'client' in turn_files:
                ordered_files.append(turn_files['client'])
            
            # Then add CSR audio
            if 'csr' in turn_files:
                ordered_files.append(turn_files['csr'])
        
        if not ordered_files:
            logger.error("No valid audio files found")
            return None
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"conversation_complete_{conversation_id}_{timestamp}.wav"
        output_path = self.output_dir / output_filename
        
        # Merge audio files
        success = self.merge_wav_files(ordered_files, str(output_path), silence_duration)
        
        if success:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Merged conversation audio saved: {output_filename} ({file_size_mb:.1f}MB)")
            return str(output_path)
        else:
            logger.error("Failed to merge conversation audio")
            return None

def merge_latest_conversation_simple(output_dir: str = "output", silence_duration: float = 0.5) -> Optional[Dict[str, str]]:
    """
    Convenience function to merge the latest conversation audio using simple wave merger.
    
    Args:
        output_dir: Directory containing audio files
        silence_duration: Seconds of silence between turns
        
    Returns:
        Dictionary with paths to merged audio or None if failed
    """
    merger = SimpleAudioMerger(output_dir)
    
    try:
        merged_audio_path = merger.merge_conversation_audio(silence_duration=silence_duration)
        
        if merged_audio_path:
            conversation_id = merger.find_latest_conversation()
            return {
                "merged_audio": merged_audio_path,
                "conversation_id": conversation_id or "unknown"
            }
        else:
            return None
        
    except Exception as e:
        logger.error(f"Failed to merge conversation audio: {e}")
        return None

if __name__ == "__main__":
    # Test the simple merger
    logging.basicConfig(level=logging.INFO)
    
    try:
        result = merge_latest_conversation_simple()
        if result:
            print(f"SUCCESS: Conversation audio merged!")
            print(f"Merged audio: {result['merged_audio']}")
            print(f"Conversation ID: {result['conversation_id']}")
        else:
            print("FAILED: Could not merge conversation audio")
        
    except Exception as e:
        print(f"ERROR: {e}")
