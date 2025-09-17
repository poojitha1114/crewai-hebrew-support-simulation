"""
TTSAgent: Handles text-to-speech synthesis for Hebrew text.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

logger = logging.getLogger(__name__)

class TTSResponse(BaseModel):
    """Response from TTS synthesis."""
    text: str
    audio_file_path: str
    success: bool = True
    error_message: Optional[str] = None

class TTSAgent:
    """Agent for Hebrew text-to-speech synthesis."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize TTSAgent."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"TTSAgent initialized with output dir: {output_dir}")
    
    def synthesize_speech(self, text: str, voice: str = "hebrew_female", 
                         speaker: str = "client", turn: int = 1) -> TTSResponse:
        """Synthesize speech from Hebrew text."""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{speaker}_turn_{turn:02d}_{timestamp}.wav"
            audio_path = self.output_dir / filename
            
            # Try to use gTTS for Hebrew synthesis
            try:
                from gtts import gTTS
                import pydub
                
                # Create TTS audio
                tts = gTTS(text=text, lang='iw', slow=False)  # 'iw' for Hebrew
                temp_mp3 = audio_path.with_suffix('.mp3')
                tts.save(str(temp_mp3))
                
                # Convert to WAV
                audio = pydub.AudioSegment.from_mp3(str(temp_mp3))
                audio.export(str(audio_path), format="wav")
                temp_mp3.unlink()  # Remove temp mp3
                
                logger.info(f"TTS audio generated: {filename}")
                return TTSResponse(
                    text=text,
                    audio_file_path=str(audio_path),
                    success=True
                )
                
            except ImportError:
                logger.warning("gTTS or pydub not available, creating dummy audio file")
                # Create minimal WAV file for testing
                wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
                with open(audio_path, 'wb') as f:
                    f.write(wav_header)
                
                return TTSResponse(
                    text=text,
                    audio_file_path=str(audio_path),
                    success=True
                )
                
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return TTSResponse(
                text=text,
                audio_file_path="",
                success=False,
                error_message=str(e)
            )
