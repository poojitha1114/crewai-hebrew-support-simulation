"""
Merge Conversation Audio - Standalone script to merge individual audio files into complete conversation.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.conversation_audio_merger import merge_latest_conversation, ConversationAudioMerger

def main():
    """Main function to merge conversation audio."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Hebrew Conversation Audio Merger")
    print("=" * 40)
    
    try:
        # Check if specific conversation ID provided
        if len(sys.argv) > 1:
            conversation_id = sys.argv[1]
            print(f"Merging specific conversation: {conversation_id}")
            
            merger = ConversationAudioMerger()
            merged_audio_path = merger.merge_conversation_audio(conversation_id)
            summary_path = merger.create_conversation_summary(conversation_id, merged_audio_path)
            
            result = {
                "merged_audio": merged_audio_path,
                "summary": summary_path,
                "conversation_id": conversation_id
            }
        else:
            print("Merging latest conversation...")
            result = merge_latest_conversation()
        
        print("\nSUCCESS: Conversation audio merged!")
        print(f"Merged audio file: {Path(result['merged_audio']).name}")
        print(f"Summary file: {Path(result['summary']).name}")
        print(f"Conversation ID: {result['conversation_id']}")
        
        # Display file info
        audio_path = Path(result['merged_audio'])
        if audio_path.exists():
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"File size: {size_mb:.1f} MB")
        
        print(f"\nPlay the merged audio:")
        print(f"   start \"{result['merged_audio']}\"")
        
        return True
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages: pip install pydub")
        return False
        
    except Exception as e:
        print(f"Error merging audio: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
