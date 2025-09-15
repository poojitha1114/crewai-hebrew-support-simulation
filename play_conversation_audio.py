"""
Simple Conversation  Audio Player - Direct audio playback without webview dependencies.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConversationAudioPlayer:
    """Simple audio player for conversation files."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the audio player."""
        self.output_dir = Path(output_dir)
        
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
    
    def find_merged_audio_files(self):
        """Find all merged conversation audio files."""
        merged_files = []
        
        for audio_file in self.output_dir.glob("conversation_complete_*.wav"):
            try:
                # Get file info
                stat = audio_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)
                
                merged_files.append({
                    "path": str(audio_file),
                    "name": audio_file.name,
                    "size_mb": size_mb,
                    "modified": modified
                })
            except Exception as e:
                logger.warning(f"Error reading file {audio_file}: {e}")
        
        # Sort by modification time (newest first)
        merged_files.sort(key=lambda x: x["modified"], reverse=True)
        return merged_files
    
    def play_audio_file(self, file_path: str):
        """Play an audio file using pygame."""
        if not PYGAME_AVAILABLE:
            print("pygame not available. Install with: pip install pygame")
            return False
        
        try:
            if not os.path.exists(file_path):
                print(f"Audio file not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size < 1000:
                print(f"Audio file too small ({file_size} bytes), likely corrupted")
                return False
            
            print(f"Loading audio: {Path(file_path).name}")
            pygame.mixer.music.load(file_path)
            
            print("Playing audio... Press Ctrl+C to stop")
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            print("Playback completed!")
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def play_with_system_player(self, file_path: str):
        """Play audio using system default player."""
        try:
            if not os.path.exists(file_path):
                print(f"Audio file not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size < 1000:
                print(f"Audio file too small ({file_size} bytes), likely corrupted")
                return False
            
            print(f"Opening with system player: {Path(file_path).name}")
            
            # Use Windows start command
            os.system(f'start "" "{file_path}"')
            return True
            
        except Exception as e:
            print(f"Error opening with system player: {e}")
            return False

def main():
    """Main function to play conversation audio."""
    logging.basicConfig(level=logging.INFO)
    
    print("Hebrew Conversation Audio Player")
    print("=" * 40)
    
    player = ConversationAudioPlayer()
    
    # Find all merged audio files
    merged_files = player.find_merged_audio_files()
    
    if not merged_files:
        print("No merged conversation audio files found.")
        print("Run 'python merge_conversation_audio.py' first to create merged audio.")
        return
    
    print(f"Found {len(merged_files)} merged conversation audio files:\n")
    
    # Display available files
    for i, file_info in enumerate(merged_files, 1):
        print(f"{i}. {file_info['name']}")
        print(f"   Size: {file_info['size_mb']:.1f} MB")
        print(f"   Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Get user choice
    try:
        if len(merged_files) == 1:
            choice = 1
            print(f"Playing the only available file...")
        else:
            choice = int(input(f"Select file to play (1-{len(merged_files)}): "))
        
        if 1 <= choice <= len(merged_files):
            selected_file = merged_files[choice - 1]
            file_path = selected_file["path"]
            
            print(f"\nSelected: {selected_file['name']}")
            
            # Try pygame first, then system player
            print("\nChoose playback method:")
            print("1. Pygame (in-terminal)")
            print("2. System player (opens external app)")
            
            method = input("Enter choice (1 or 2, default=2): ").strip()
            
            if method == "1" and PYGAME_AVAILABLE:
                success = player.play_audio_file(file_path)
            else:
                success = player.play_with_system_player(file_path)
            
            if not success:
                print("Playback failed. Try the other method or check the audio file.")
        else:
            print("Invalid selection.")
    
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
