"""
Main orchestration script for Hebrew customer service call simulation.
Uses CrewAI Flow to coordinate all agents in the conversation loop.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from crewai import Flow, Crew, Task
from dotenv import load_dotenv

# Import all agents
from agents.nikud_agent import NikudAgent
from agents.tts_agent import TTSAgent
from agents.stt_agent import STTAgent
from agents.client_agent import ClientAgent
from agents.csr_agent import CSRAgent
from agents.transcript_agent import TranscriptAgent
from agents.token_monitor_agent import TokenMonitorAgent, TokenBudget
from agents.guardrail_agent import GuardrailAgent, GuardrailConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/conversation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HebrewCustomerServiceFlow(Flow):
    """CrewAI Flow for Hebrew customer service call simulation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the conversation flow.
        
        Args:
            config: Configuration dictionary with service URLs and settings
        """
        super().__init__()
        self.config = config
        self.conversation_active = False
        self.max_turns = config.get('max_turns', 15)
        
        # Initialize all agents
        self._initialize_agents()
        
        # Conversation state
        self.turn_counter = 0
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _initialize_agents(self):
        """Initialize all agents with configuration."""
        logger.info("Initializing agents...")
        
        # Core processing agents
        self.nikud_agent = NikudAgent(
            phonikud_url=self.config.get('phonikud_url', 'http://localhost:8000')
        )
        
        self.tts_agent = TTSAgent()
        
        self.stt_agent = STTAgent(
            whisper_url=self.config.get('whisper_url'),
            local_mode=self.config.get('whisper_local', True),
            model_size=self.config.get('whisper_model', 'base')
        )
        
        # Conversation agents
        self.client_agent = ClientAgent(
            nikud_agent=self.nikud_agent,
            tts_agent=self.tts_agent,
            personality=self.config.get('client_personality', 'polite_but_determined')
        )
        
        self.csr_agent = CSRAgent(
            model_config=self.config.get('llm_config', {
                'primary_model': 'gpt-4o-mini',
                'fallback_models': ['gpt-3.5-turbo'],
                'temperature': 0.7,
                'max_tokens': 500
            })
        )
        
        # Management agents
        self.transcript_agent = TranscriptAgent(
            output_dir=self.config.get('output_dir', 'output')
        )
        
        self.token_monitor = TokenMonitorAgent(
            budget_config=TokenBudget(
                max_tokens_per_conversation=self.config.get('max_tokens', 10000),
                warning_threshold=0.8,
                summarization_threshold=0.9
            )
        )
        
        self.guardrail_agent = GuardrailAgent(
            config=GuardrailConfig(
                max_retries=3,
                min_stt_confidence=0.6,
                max_conversation_turns=self.max_turns
            )
        )
        
        logger.info("All agents initialized successfully")
    
    def start_conversation(self) -> str:
        """Start a new conversation."""
        try:
            # Reset all agent states
            self.client_agent.reset_state()
            self.csr_agent.reset_state()
            self.token_monitor.reset_conversation_usage()
            
            # Start transcript
            self.transcript_agent.start_conversation(
                self.conversation_id,
                self.client_agent.personality
            )
            
            self.conversation_active = True
            self.turn_counter = 0
            
            logger.info(f"Started conversation: {self.conversation_id}")
            return self.conversation_id
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise
    
    def conversation_turn(self, csr_message: str = "") -> Dict:
        """Execute one turn of the conversation."""
        try:
            self.turn_counter += 1
            logger.info(f"=== Turn {self.turn_counter} ===")
            
            # Check guardrails for turn limit
            guardrail_check = self.guardrail_agent.check_content(
                "", "turn_check", "system", self.turn_counter
            )
            
            if guardrail_check.conversation_should_end:
                logger.info("Conversation ended by guardrails")
                return self._end_conversation("guardrail_limit")
            
            # Generate client response
            client_response = self.client_agent.generate_response(
                csr_message, 
                self.transcript_agent.get_transcript_for_context(5)
            )
            
            if not client_response.success:
                logger.error(f"Client response failed: {client_response.error_message}")
                return self._end_conversation("client_response_failed")
            
            # Check client response with guardrails
            client_guardrail = self.guardrail_agent.check_content(
                client_response.text, "client_text", "client", self.turn_counter
            )
            
            if not client_guardrail.allowed:
                logger.warning("Client response blocked by guardrails")
                # Use modified content if available
                if client_guardrail.modified_content:
                    client_response.text = client_guardrail.modified_content
                else:
                    return {"success": False, "error": "client_blocked"}
            
            # Add client entry to transcript
            self.transcript_agent.add_entry(
                turn=self.turn_counter,
                speaker="client",
                text=client_response.text,
                nikud_text=client_response.nikud_text,
                audio_file_path=client_response.audio_file_path,
                confidence=1.0,  # Client generated, so full confidence
                processing_time=0.0
            )
            
            # Check if client is satisfied (conversation should end)
            if client_response.state.satisfied:
                logger.info("Client satisfied, ending conversation")
                return self._end_conversation("client_satisfied")
            
            # Generate CSR response
            transcript_summary = self.transcript_agent.get_transcript_for_context(5)
            csr_response = self.csr_agent.generate_response(
                client_response.text,
                transcript_summary
            )
            
            if not csr_response.success:
                logger.error(f"CSR response failed: {csr_response.error_message}")
                return {"success": False, "error": "csr_response_failed"}
            
            # Update token usage
            token_status = self.token_monitor.record_usage(
                prompt_tokens=csr_response.tokens_used // 2,  # Estimate
                completion_tokens=csr_response.tokens_used // 2,
                total_tokens=csr_response.tokens_used
            )
            
            # Check if summarization is needed
            if token_status.action_required == "summarize":
                logger.info("Triggering conversation summarization")
                self.token_monitor.trigger_summarization()
            
            # Check CSR response with guardrails
            csr_guardrail = self.guardrail_agent.check_content(
                csr_response.text, "csr_text", "csr", self.turn_counter
            )
            
            if not csr_guardrail.allowed:
                logger.warning("CSR response blocked by guardrails")
                if csr_guardrail.modified_content:
                    csr_response.text = csr_guardrail.modified_content
                else:
                    return {"success": False, "error": "csr_blocked"}
            
            # Generate CSR audio using TTS
            csr_audio_response = self.tts_agent.synthesize_speech(
                text=csr_response.text,
                voice="hebrew_female",
                speaker="csr",
                turn=self.turn_counter
            )
            
            csr_audio_file = ""
            if csr_audio_response.success:
                csr_audio_file = csr_audio_response.audio_file_path
                logger.info(f"CSR audio generated: {csr_audio_file}")
            else:
                logger.warning(f"CSR audio generation failed: {csr_audio_response.error_message}")
            
            # Add CSR entry to transcript
            self.transcript_agent.add_entry(
                turn=self.turn_counter,
                speaker="csr",
                text=csr_response.text,
                audio_file_path=csr_audio_file,
                confidence=1.0,
                processing_time=0.0
            )
            
            # Update transcript with token usage
            self.transcript_agent.update_tokens_used(csr_response.tokens_used)
            
            # Check if cancellation was approved
            if csr_response.state.cancellation_approved:
                logger.info("Cancellation approved, ending conversation")
                return self._end_conversation("cancellation_approved")
            
            # Return turn results
            return {
                "success": True,
                "turn": self.turn_counter,
                "client_response": {
                    "text": client_response.text,
                    "audio_file": client_response.audio_file_path,
                    "state": client_response.state.dict()
                },
                "csr_response": {
                    "text": csr_response.text,
                    "action": csr_response.action,
                    "state": csr_response.state.dict()
                },
                "token_usage": token_status.current_usage.dict(),
                "guardrail_status": {
                    "client_violations": len(client_guardrail.violations),
                    "csr_violations": len(csr_guardrail.violations)
                },
                "conversation_active": self.conversation_active
            }
            
        except Exception as e:
            logger.error(f"Conversation turn failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _end_conversation(self, outcome: str) -> Dict:
        """End the current conversation."""
        try:
            self.conversation_active = False
            
            # Save transcript
            transcript_path = self.transcript_agent.end_conversation(outcome)
            
            # Get final statistics
            stats = self._get_conversation_stats()
            
            logger.info(f"Conversation ended: {outcome}")
            logger.info(f"Transcript saved: {transcript_path}")
            
            return {
                "success": True,
                "conversation_ended": True,
                "outcome": outcome,
                "transcript_path": transcript_path,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            return {"success": False, "error": str(e)}
    
    def run_full_conversation(self) -> Dict:
        """Run a complete conversation from start to finish."""
        try:
            # Start conversation
            conversation_id = self.start_conversation()
            
            # Initial client message (greeting)
            result = self.conversation_turn()
            if not result.get("success"):
                return result
            
            # Continue conversation loop
            while self.conversation_active and self.turn_counter < self.max_turns:
                # Get last CSR response for next client turn
                last_csr_response = result.get("csr_response", {}).get("text", "")
                
                # Next turn
                result = self.conversation_turn(last_csr_response)
                if not result.get("success"):
                    break
                
                # Check if conversation ended
                if not result.get("conversation_active", True):
                    break
            
            # End conversation if still active
            if self.conversation_active:
                result = self._end_conversation("max_turns_reached")
            
            return result
            
        except Exception as e:
            logger.error(f"Full conversation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_conversation_stats(self) -> Dict:
        """Get comprehensive conversation statistics."""
        return {
            "conversation_id": self.conversation_id,
            "total_turns": self.turn_counter,
            "client_stats": self.client_agent.get_stats(),
            "csr_stats": self.csr_agent.get_stats(),
            "token_stats": self.token_monitor.get_stats(),
            "guardrail_stats": self.guardrail_agent.get_stats(),
            "transcript_stats": self.transcript_agent.get_stats()
        }

def load_configuration() -> Dict:
    """Load configuration from environment and defaults."""
    load_dotenv()
    
    return {
        # Service URLs
        'phonikud_url': os.getenv('PHONIKUD_URL', 'http://localhost:8000'),
        'chatterbox_url': os.getenv('CHATTERBOX_URL', 'http://localhost:8001'),
        'whisper_url': os.getenv('WHISPER_URL', 'http://localhost:8002'),
        'whisper_local': os.getenv('WHISPER_LOCAL', 'true').lower() == 'true',
        'whisper_model': os.getenv('WHISPER_MODEL', 'base'),
        
        # LLM Configuration
        'llm_config': {
            'primary_model': os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
            'fallback_models': os.getenv('FALLBACK_MODELS', 'gpt-3.5-turbo').split(','),
            'temperature': float(os.getenv('TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('MAX_TOKENS', '500'))
        },
        
        # Conversation settings
        'max_turns': int(os.getenv('MAX_TURNS', '15')),
        'max_tokens': int(os.getenv('MAX_CONVERSATION_TOKENS', '10000')),
        'client_personality': os.getenv('CLIENT_PERSONALITY', 'polite_but_determined'),
        'output_dir': os.getenv('OUTPUT_DIR', 'output')
    }

def run_demo():
    """Run a demonstration conversation."""
    logger.info("Starting Hebrew Customer Service Call Simulation Demo")
    
    try:
        # Load configuration
        config = load_configuration()
        
        # Create and run conversation flow
        flow = HebrewCustomerServiceFlow(config)
        result = flow.run_full_conversation()
        
        if result.get("success"):
            logger.info("Demo completed successfully!")
            logger.info(f"Outcome: {result.get('outcome')}")
            logger.info(f"Total turns: {result.get('stats', {}).get('total_turns', 0)}")
            logger.info(f"Transcript: {result.get('transcript_path')}")
            
            # Calculate conversation timing
            stats = result.get('stats', {})
            start_time = datetime.now()
            
            # Print comprehensive summary
            print("\n" + "="*60)
            print("🎯 HEBREW CUSTOMER SERVICE CONVERSATION SUMMARY")
            print("="*60)
            
            # Basic conversation info
            print(f"📋 Conversation ID: {stats.get('conversation_id', 'N/A')}")
            print(f"🎭 Client Personality: {config.get('client_personality', 'N/A')}")
            print(f"🏁 Outcome: {result.get('outcome', 'N/A')}")
            print(f"🔄 Total Turns: {stats.get('total_turns', 0)}")
            
            # Token usage details
            token_stats = stats.get('token_stats', {})
            if token_stats:
                usage = token_stats.get('current_usage', {})
                budget = token_stats.get('budget', {})
                print(f"\n💰 TOKEN USAGE:")
                print(f"   Total Tokens Used: {usage.get('total_tokens_used', 0)}")
                print(f"   Prompt Tokens: {usage.get('prompt_tokens', 0)}")
                print(f"   Completion Tokens: {usage.get('completion_tokens', 0)}")
                print(f"   Budget Limit: {budget.get('max_tokens_per_conversation', 0)}")
                print(f"   Summarizations: {usage.get('summarizations_triggered', 0)}")
            
            # Agent statistics
            client_stats = stats.get('client_stats', {})
            csr_stats = stats.get('csr_stats', {})
            if client_stats or csr_stats:
                print(f"\n👥 AGENT PERFORMANCE:")
                if client_stats:
                    print(f"   Client Responses: {client_stats.get('total_responses', 0)}")
                    print(f"   Client Frustration: {client_stats.get('final_frustration', 'N/A')}")
                if csr_stats:
                    print(f"   CSR Responses: {csr_stats.get('total_responses', 0)}")
                    print(f"   Offers Made: {csr_stats.get('offers_made', 0)}")
            
            # Guardrail statistics
            guardrail_stats = stats.get('guardrail_stats', {})
            if guardrail_stats:
                print(f"\n🛡️  GUARDRAILS:")
                print(f"   Violations: {guardrail_stats.get('violation_count', 0)}")
            
            # File locations
            print(f"\n📁 FILES GENERATED:")
            print(f"   Transcript: {Path(result.get('transcript_path', '')).name}")
            
            # Audio files count
            import glob
            audio_files = glob.glob("output/*_turn_*.wav")
            print(f"   Audio Files: {len(audio_files)} individual turn files")
            
            print("="*60)
            print("⏱️  Ready for audio merging and timing...")
            
        else:
            logger.error(f"Demo failed: {result.get('error')}")
            print(f"Demo failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"Demo execution failed: {e}")

def merge_conversation_audio_post_demo():
    """Merge conversation audio files after demo completion."""
    try:
        from utils.conversation_audio_merger import merge_latest_conversation
        import time
        
        print("\n🎵 AUDIO MERGING PROCESS")
        print("="*40)
        
        # Start timing the audio merge process
        merge_start_time = time.time()
        print("⏱️  Starting audio merge timing...")
        
        result = merge_latest_conversation()
        
        # Calculate merge duration
        merge_end_time = time.time()
        merge_duration = merge_end_time - merge_start_time
        
        print(f"\n✅ AUDIO MERGE COMPLETED:")
        print(f"   📄 File: {Path(result['merged_audio']).name}")
        print(f"   ⏱️  Merge Process Time: {merge_duration:.2f} seconds")
        print(f"   📊 Summary: {Path(result['summary']).name}")
        
        # Read and display audio statistics
        try:
            import json
            with open(result['summary'], 'r', encoding='utf-8') as f:
                audio_stats = json.load(f)
            
            stats = audio_stats.get('statistics', {})
            print(f"\n🎧 AUDIO STATISTICS:")
            print(f"   Duration: {stats.get('duration_formatted', 'N/A')} ({stats.get('duration_seconds', 0):.1f}s)")
            print(f"   Total Turns: {stats.get('total_turns', 0)}")
            print(f"   Client Turns: {stats.get('client_turns', 0)}")
            print(f"   CSR Turns: {stats.get('csr_turns', 0)}")
            
        except Exception as e:
            logger.warning(f"Could not read audio statistics: {e}")
        
        print("="*40)
        logger.info(f"Merged conversation audio: {result['merged_audio']}")
        
        return result['merged_audio']
        
    except ImportError:
        print("❌ Audio merging requires pydub: pip install pydub")
        logger.warning("pydub not available for audio merging")
        return None
    except Exception as e:
        print(f"❌ Audio merging failed: {e}")
        logger.warning(f"Audio merging failed: {e}")
        return None

if __name__ == "__main__":
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    # Run demo
    run_demo()
    
    # Merge conversation audio after demo
    merge_conversation_audio_post_demo()
