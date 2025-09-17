"""
TranscriptAgent: Manages conversation transcripts and logging.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TranscriptEntry(BaseModel):
    """Single entry in conversation transcript."""
    turn: int
    speaker: str
    text: str
    nikud_text: Optional[str] = None
    audio_file_path: Optional[str] = None
    confidence: float = 1.0
    processing_time: float = 0.0
    timestamp: str

class TranscriptAgent:
    """Agent for managing conversation transcripts."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize TranscriptAgent."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.conversation_id = None
        self.entries = []
        self.tokens_used = 0
        logger.info(f"TranscriptAgent initialized with output dir: {output_dir}")
    
    def start_conversation(self, conversation_id: str, client_personality: str):
        """Start a new conversation transcript."""
        self.conversation_id = conversation_id
        self.entries = []
        self.tokens_used = 0
        self.client_personality = client_personality
        logger.info(f"Started transcript for conversation: {conversation_id}")
    
    def add_entry(self, turn: int, speaker: str, text: str, 
                  nikud_text: Optional[str] = None, 
                  audio_file_path: Optional[str] = None,
                  confidence: float = 1.0, processing_time: float = 0.0):
        """Add entry to transcript."""
        entry = TranscriptEntry(
            turn=turn,
            speaker=speaker,
            text=text,
            nikud_text=nikud_text,
            audio_file_path=audio_file_path,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        self.entries.append(entry)
        logger.info(f"Added transcript entry - Turn {turn}, Speaker: {speaker}")
    
    def update_tokens_used(self, tokens: int):
        """Update token usage count."""
        self.tokens_used += tokens
    
    def get_transcript_for_context(self, last_n_turns: int = 5) -> str:
        """Get recent transcript entries for context."""
        recent_entries = self.entries[-last_n_turns*2:] if self.entries else []
        context = []
        for entry in recent_entries:
            context.append(f"{entry.speaker}: {entry.text}")
        return "\n".join(context)
    
    def end_conversation(self, outcome: str) -> str:
        """End conversation and save transcript."""
        transcript_data = {
            "metadata": {
                "conversation_id": self.conversation_id,
                "outcome": outcome,
                "total_turns": len([e for e in self.entries if e.speaker == "client"]),
                "tokens_used": self.tokens_used,
                "client_personality": getattr(self, 'client_personality', 'unknown'),
                "start_time": self.entries[0].timestamp if self.entries else None,
                "end_time": datetime.now().isoformat()
            },
            "entries": [entry.dict() for entry in self.entries]
        }
        
        # Save transcript
        transcript_path = self.output_dir / f"transcript_{self.conversation_id}.json"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Transcript saved: {transcript_path}")
        return str(transcript_path)
    
    def get_stats(self) -> Dict:
        """Get transcript statistics."""
        return {
            "total_entries": len(self.entries),
            "tokens_used": self.tokens_used,
            "conversation_id": self.conversation_id
        }
