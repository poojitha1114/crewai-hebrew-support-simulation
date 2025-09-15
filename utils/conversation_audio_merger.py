"""
Conversation Audio Merger - Merges individual turn audio files into complete conversation.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConversationAudioMerger:
    """Merges individual conversation turn audio files into a complete conversation audio."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the audio merger.
        
        Args:
            output_dir: Directory containing audio files and transcripts
        """
        self.output_dir = Path(output_dir)
        
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub is required for audio merging. Install with: pip install pydub")
    
    def find_conversation_files(self, conversation_id: str) -> Dict[int, Dict[str, str]]:
        """
        Find all audio files for a specific conversation.
        
        Args:
            conversation_id: The conversation ID to find files for
            
        Returns:
            Dictionary mapping turn numbers to client/csr audio files
        """
        audio_files = {}
        
        # Extract date and approximate time from conversation_id (format: conv_YYYYMMDD_HHMMSS)
        if conversation_id.startswith("conv_"):
            timestamp = conversation_id[5:]  # Remove "conv_" prefix
        else:
            timestamp = conversation_id
        
        # Extract date part (YYYYMMDD) and hour/minute (HHMM) for flexible matching
        if len(timestamp) >= 13:  # 20250913_162328
            date_part = timestamp[:8]  # 20250913
            time_part = timestamp[9:13]  # 1623 (hour and minute)
        else:
            date_part = timestamp
            time_part = ""
        
        logger.info(f"Looking for audio files from date {date_part}, time around {time_part}")
        
        # Find all audio files matching this conversation
        for audio_file in self.output_dir.glob("*.wav"):
            filename = audio_file.name
            
            # Skip merged conversation files
            if filename.startswith("conversation_complete_"):
                continue
            
            # Check if file belongs to this conversation by date and approximate time
            # For files like: client_turn_001_20250913_162332.wav
            # We match by date (20250913) and similar time (162X)
            if date_part in filename:
                parts = filename.split('_')
                if len(parts) >= 5:  # speaker_turn_number_date_time.wav
                    speaker = parts[0]  # client or csr
                    turn_str = parts[2]  # turn number (with leading zeros)
                    file_date = parts[3]  # 20250913
                    file_time = parts[4].replace('.wav', '')  # 162332
                    
                    # Check if it's the same date and similar time (within same hour/minute range)
                    if (file_date == date_part and 
                        (not time_part or file_time.startswith(time_part))):
                        
                        if turn_str.isdigit():
                            turn_num = int(turn_str)
                            
                            if turn_num not in audio_files:
                                audio_files[turn_num] = {}
                            
                            audio_files[turn_num][speaker] = str(audio_file)
                            logger.info(f"Found audio file: {filename} -> Turn {turn_num}, Speaker {speaker}")
        
        logger.info(f"Found {len(audio_files)} turns with audio files")
        return audio_files
    
    def _create_tone_marker(self, frequency: int, duration_ms: int) -> AudioSegment:
        """Create a simple tone marker for speaker identification."""
        try:
            # Generate a simple sine wave tone
            import math
            sample_rate = 22050
            duration_seconds = duration_ms / 1000.0
            
            # Generate sine wave samples
            samples = []
            for i in range(int(sample_rate * duration_seconds)):
                t = i / sample_rate
                sample = int(16383 * math.sin(2 * math.pi * frequency * t))
                samples.extend([sample, sample])  # Stereo
            
            # Convert to AudioSegment
            audio_data = bytes()
            for sample in samples:
                # Convert to 16-bit signed integer, little-endian
                audio_data += sample.to_bytes(2, byteorder='little', signed=True)
            
            tone = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=sample_rate,
                channels=2  # Stereo
            )
            
            # Fade in/out to avoid clicks
            tone = tone.fade_in(20).fade_out(20)
            return tone
            
        except Exception as e:
            logger.warning(f"Failed to create tone marker: {e}")
            # Return silence if tone generation fails
            return AudioSegment.silent(duration=duration_ms)
    
    def find_latest_conversation(self) -> Optional[str]:
        """
        Find the most recent conversation ID from transcript files.
        
        Returns:
            Latest conversation ID or None if no transcripts found
        """
        transcript_files = list(self.output_dir.glob("transcript_*.json"))
        
        if not transcript_files:
            return None
        
        # Sort by modification time, get latest
        latest_transcript = max(transcript_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_transcript, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('metadata', {}).get('conversation_id')
        except Exception as e:
            logger.error(f"Failed to read transcript {latest_transcript}: {e}")
            return None
    
    def merge_conversation_audio(self, conversation_id: str = None, 
                               silence_duration: float = 0.5) -> str:
        """
        Merge all audio files from a conversation into one complete audio file.
        
        Args:
            conversation_id: Specific conversation to merge (if None, uses latest)
            silence_duration: Seconds of silence between turns
            
        Returns:
            Path to the merged audio file
        """
        if conversation_id is None:
            conversation_id = self.find_latest_conversation()
            if conversation_id is None:
                raise ValueError("No conversations found")
        
        logger.info(f"Merging audio for conversation: {conversation_id}")
        
        # Find all audio files for this conversation
        audio_files = self.find_conversation_files(conversation_id)
        
        if not audio_files:
            raise ValueError(f"No audio files found for conversation {conversation_id}")
        
        # Create silence segment and audio markers
        silence = AudioSegment.silent(duration=int(silence_duration * 1000))  # pydub uses milliseconds
        
        # Create simple tone markers to identify speakers
        # Client marker: single beep (440Hz for 0.2s)
        client_marker = self._create_tone_marker(440, 200)  # A4 note
        # CSR marker: double beep (523Hz for 0.15s each)
        csr_marker = self._create_tone_marker(523, 150) + AudioSegment.silent(duration=50) + self._create_tone_marker(523, 150)  # C5 note
        
        # Merge audio files in turn order
        merged_audio = AudioSegment.empty()
        
        for turn_num in sorted(audio_files.keys()):
            turn_files = audio_files[turn_num]
            
            # Add client audio if available
            if 'client' in turn_files and os.path.exists(turn_files['client']):
                try:
                    # Check file size first - corrupted files are often very small
                    file_size = os.path.getsize(turn_files['client'])
                    if file_size < 1000:  # Less than 1KB is likely corrupted
                        logger.warning(f"Skipping client turn {turn_num}: file too small ({file_size} bytes)")
                        continue
                    
                    # Try different audio formats in case of encoding issues
                    client_audio = None
                    try:
                        client_audio = AudioSegment.from_wav(turn_files['client'])
                    except:
                        try:
                            client_audio = AudioSegment.from_file(turn_files['client'])
                        except:
                            logger.warning(f"Skipping client turn {turn_num}: unable to decode audio file")
                            continue
                    
                    if client_audio and len(client_audio) > 0:
                        merged_audio += client_marker + AudioSegment.silent(duration=100) + client_audio + silence
                        logger.info(f"Added client turn {turn_num}: {Path(turn_files['client']).name} ({len(client_audio)/1000:.1f}s)")
                    else:
                        logger.warning(f"Skipping client turn {turn_num}: empty audio")
                        
                except Exception as e:
                    logger.warning(f"Failed to load client audio for turn {turn_num}: {e}")
            
            # Add CSR audio if available
            if 'csr' in turn_files and os.path.exists(turn_files['csr']):
                try:
                    # Check file size first - corrupted files are often very small
                    file_size = os.path.getsize(turn_files['csr'])
                    if file_size < 1000:  # Less than 1KB is likely corrupted
                        logger.warning(f"Skipping CSR turn {turn_num}: file too small ({file_size} bytes)")
                        continue
                    
                    # Try different audio formats in case of encoding issues
                    csr_audio = None
                    try:
                        csr_audio = AudioSegment.from_wav(turn_files['csr'])
                    except:
                        try:
                            csr_audio = AudioSegment.from_file(turn_files['csr'])
                        except:
                            logger.warning(f"Skipping CSR turn {turn_num}: unable to decode audio file")
                            continue
                    
                    if csr_audio and len(csr_audio) > 0:
                        merged_audio += csr_marker + AudioSegment.silent(duration=100) + csr_audio + silence
                        logger.info(f"Added CSR turn {turn_num}: {Path(turn_files['csr']).name} ({len(csr_audio)/1000:.1f}s)")
                    else:
                        logger.warning(f"Skipping CSR turn {turn_num}: empty audio")
                        
                except Exception as e:
                    logger.warning(f"Failed to load CSR audio for turn {turn_num}: {e}")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"conversation_complete_{conversation_id}_{timestamp}.wav"
        output_path = self.output_dir / output_filename
        
        # Export merged audio
        merged_audio.export(str(output_path), format="wav")
        
        duration_seconds = len(merged_audio) / 1000.0
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"Merged conversation audio saved: {output_filename}")
        logger.info(f"Duration: {duration_seconds:.1f}s, Size: {file_size_mb:.1f}MB")
        
        return str(output_path)
    
    def create_conversation_summary(self, conversation_id: str, merged_audio_path: str) -> str:
        """
        Create a summary file for the merged conversation.
        
        Args:
            conversation_id: The conversation ID
            merged_audio_path: Path to the merged audio file
            
        Returns:
            Path to the summary file
        """
        audio_files = self.find_conversation_files(conversation_id)
        
        # Calculate statistics
        total_turns = len(audio_files)
        client_turns = sum(1 for turn in audio_files.values() if 'client' in turn)
        csr_turns = sum(1 for turn in audio_files.values() if 'csr' in turn)
        
        # Get audio duration
        if PYDUB_AVAILABLE and os.path.exists(merged_audio_path):
            try:
                audio = AudioSegment.from_wav(merged_audio_path)
                duration_seconds = len(audio) / 1000.0
            except:
                duration_seconds = 0
        else:
            duration_seconds = 0
        
        summary = {
            "conversation_id": conversation_id,
            "merged_audio_file": Path(merged_audio_path).name,
            "created_at": datetime.now().isoformat(),
            "statistics": {
                "total_turns": total_turns,
                "client_turns": client_turns,
                "csr_turns": csr_turns,
                "duration_seconds": duration_seconds,
                "duration_formatted": f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}"
            },
            "individual_files": audio_files
        }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"conversation_audio_summary_{conversation_id}_{timestamp}.json"
        summary_path = self.output_dir / summary_filename
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Audio summary saved: {summary_filename}")
        return str(summary_path)

def merge_latest_conversation(output_dir: str = "output", silence_duration: float = 0.5) -> Dict[str, str]:
    """
    Convenience function to merge the latest conversation audio.
    
    Args:
        output_dir: Directory containing audio files
        silence_duration: Seconds of silence between turns
        
    Returns:
        Dictionary with paths to merged audio and summary files
    """
    merger = ConversationAudioMerger(output_dir)
    
    try:
        # Merge audio
        merged_audio_path = merger.merge_conversation_audio(silence_duration=silence_duration)
        
        # Get conversation ID from the merged filename
        filename = Path(merged_audio_path).name
        # Extract conversation_id from the merged filename
        parts = filename.split('_')
        if len(parts) >= 4:
            # For conversation_complete_conv_20250913_162328_20250915_225338.wav
            # We want: conv_20250913_162328
            conversation_id = '_'.join(parts[2:5])  # conv_YYYYMMDD_HHMMSS
        else:
            conversation_id = "unknown"
        
        # Create summary
        summary_path = merger.create_conversation_summary(conversation_id, merged_audio_path)
        
        return {
            "merged_audio": merged_audio_path,
            "summary": summary_path,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to merge conversation audio: {e}")
        raise

if __name__ == "__main__":
    # Test the merger
    logging.basicConfig(level=logging.INFO)
    
    try:
        result = merge_latest_conversation()
        print(f"✅ Conversation audio merged successfully!")
        print(f"📁 Merged audio: {result['merged_audio']}")
        print(f"📄 Summary: {result['summary']}")
        print(f"🆔 Conversation ID: {result['conversation_id']}")
        
    except Exception as e:
        print(f"❌ Failed to merge audio: {e}")
