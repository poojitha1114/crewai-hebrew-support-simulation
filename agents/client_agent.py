"""
ClientAgent: Simulates a customer trying to cancel their TV subscription.
Uses NikudAgent and TTSAgent to produce Hebrew speech. 
"""

import logging
import random
from typing import Dict, List, Optional
from crewai import Agent, Task
from pydantic import BaseModel, Field
from agents.nikud_agent import NikudAgent, NikudResponse
from agents.tts_agent import TTSAgent, TTSResponse

logger = logging.getLogger(__name__)

class ClientState(BaseModel):
    """Current state of the client conversation."""
    intent: str = Field(default="cancel_subscription", description="Current intent")
    frustration_level: int = Field(default=1, ge=1, le=5, description="Frustration level 1-5")
    attempts: int = Field(default=0, ge=0, description="Number of attempts made")
    satisfied: bool = Field(default=False, description="Whether client is satisfied")
    
class ClientResponse(BaseModel):
    """Response from client agent."""
    text: str
    nikud_text: str
    audio_file_path: str
    state: ClientState
    turn: int
    success: bool = Field(default=True)
    error_message: Optional[str] = None

class ClientAgent:
    """Agent that simulates a Hebrew-speaking customer trying to cancel TV subscription."""
    
    def __init__(self, nikud_agent: NikudAgent, tts_agent: TTSAgent, 
                 personality: str = "polite_but_determined"):
        """
        Initialize ClientAgent.
        
        Args:
            nikud_agent: NikudAgent instance for text processing
            tts_agent: TTSAgent instance for speech synthesis
            personality: Client personality type
        """
        self.nikud_agent = nikud_agent
        self.tts_agent = tts_agent
        self.personality = personality
        self.state = ClientState()
        self.turn_counter = 0
        self.agent = self._create_agent()
        
        # Hebrew phrases for different scenarios
        self.phrases = self._load_hebrew_phrases()
        
    def _create_agent(self) -> Agent:
        """Create CrewAI agent for client simulation."""
        return Agent(
            role="Hebrew-Speaking TV Subscription Customer",
            goal="Successfully cancel TV subscription while maintaining realistic conversation flow",
            backstory=f"""You are a Hebrew-speaking customer who wants to cancel your TV subscription. 
            Your personality is {self.personality}. You speak naturally in Hebrew and respond 
            appropriately to customer service representatives. You have legitimate reasons for 
            canceling and will persist until you achieve your goal.""",
            verbose=True,
            allow_delegation=False
        )
    
    def _load_hebrew_phrases(self) -> Dict[str, List[str]]:
        """Load Hebrew phrases for different conversation scenarios."""
        return {
            "greeting": [
                "שלום, אני רוצה לבטל את המנוי שלי לטלוויזיה",
                "בוקר טוב, אני צריך לבטל את השירות",
                "היי, אני רוצה לסגור את החשבון שלי"
            ],
            "reason_cost": [
                "זה יקר מדי בשבילי",
                "אני לא יכול להרשות לעצמי את זה יותר",
                "המחירים עלו והתקציב שלי לא מאפשר"
            ],
            "reason_usage": [
                "אני לא משתמש בשירות מספיק",
                "אני כמעט לא צופה בטלוויזיה",
                "עברתי לנטפליקס ויוטיוב"
            ],
            "reason_technical": [
                "יש בעיות טכניות כל הזמן",
                "האיכות לא טובה",
                "השירות לא עובד כמו שצריך"
            ],
            "frustration_low": [
                "אני מבין שזה התהליך, אבל אני רוצה לבטל",
                "תודה על ההסבר, אבל ההחלטה שלי סופית",
                "אני מעריך את העזרה, אבל אני רוצה לבטל"
            ],
            "frustration_medium": [
                "אמרתי כבר שאני רוצה לבטל!",
                "למה זה כל כך מסובך לבטל?",
                "אני לא מעוניין בהצעות, רק לבטל"
            ],
            "frustration_high": [
                "זה מגוחך! אני רק רוצה לבטל!",
                "תפסיקו לנסות לשכנע אותי!",
                "אני דורש לדבר עם מנהל!"
            ],
            "confirmation": [
                "כן, אני בטוח שאני רוצה לבטל",
                "נכון, זו ההחלטה שלי",
                "בדיוק, אני רוצה לסגור את החשבון"
            ],
            "satisfaction": [
                "תודה רבה על העזרה",
                "סוף סוף! תודה",
                "מעולה, תודה לך"
            ]
        }
    
    def generate_response(self, csr_message: str = "", context: str = "") -> ClientResponse:
        """
        Generate client response based on CSR message and current state.
        
        Args:
            csr_message: Message from customer service representative
            context: Additional context for response generation
            
        Returns:
            ClientResponse with Hebrew text and audio
        """
        try:
            self.turn_counter += 1
            
            # Determine response based on state and CSR message
            response_text = self._select_response(csr_message, context)
            
            # Process text with nikud
            nikud_result = self.nikud_agent.add_nikud(response_text)
            
            # Generate audio using TTS agent with Hebrew male voice (client)
            tts_response = self.tts_agent.synthesize_speech(
                text=nikud_result.nikud_text,
                voice="hebrew_male",
                speaker="client",
                turn=self.turn_counter
            )
            
            # Update state based on response
            self._update_state(csr_message)
            
            logger.info(f"Client turn {self.turn_counter}: response generated "
                       f"(frustration: {self.state.frustration_level})")
            
            # Return success even if TTS fails (text-only mode)
            return ClientResponse(
                text=response_text,
                nikud_text=nikud_result.nikud_text,
                audio_file_path=tts_response.audio_file_path if tts_response.success else "",
                state=self.state,
                turn=self.turn_counter,
                success=True,  # Always succeed for text generation
                error_message=None
            )
            
        except Exception as e:
            error_msg = f"Client response generation failed: {e}"
            logger.error(error_msg)
            return ClientResponse(
                text="",
                nikud_text="",
                audio_file_path="",
                state=self.state,
                turn=self.turn_counter,
                success=False,
                error_message=error_msg
            )
    
    def _select_response(self, csr_message: str, context: str) -> str:
        """Select appropriate Hebrew response based on conversation state."""
        csr_lower = csr_message.lower()
        
        # Initial greeting
        if self.turn_counter == 1 or not csr_message:
            return random.choice(self.phrases["greeting"])
        
        # Handle different CSR responses
        if any(word in csr_lower for word in ["למה", "סיבה", "reason", "why"]):
            # CSR asking for reason
            reason_type = random.choice(["cost", "usage", "technical"])
            return random.choice(self.phrases[f"reason_{reason_type}"])
        
        elif any(word in csr_lower for word in ["הצעה", "הנחה", "offer", "discount"]):
            # CSR making offers
            if self.state.frustration_level <= 2:
                return random.choice(self.phrases["frustration_low"])
            elif self.state.frustration_level <= 4:
                return random.choice(self.phrases["frustration_medium"])
            else:
                return random.choice(self.phrases["frustration_high"])
        
        elif any(word in csr_lower for word in ["בטוח", "confirm", "sure", "וודא"]):
            # CSR asking for confirmation
            return random.choice(self.phrases["confirmation"])
        
        elif any(word in csr_lower for word in ["בוטל", "canceled", "סגור", "closed"]):
            # Cancellation confirmed
            self.state.satisfied = True
            return random.choice(self.phrases["satisfaction"])
        
        else:
            # Default responses based on frustration level
            if self.state.frustration_level <= 2:
                return random.choice(self.phrases["frustration_low"])
            elif self.state.frustration_level <= 4:
                return random.choice(self.phrases["frustration_medium"])
            else:
                return random.choice(self.phrases["frustration_high"])
    
    def _update_state(self, csr_message: str):
        """Update client state based on CSR interaction."""
        self.state.attempts += 1
        
        # Increase frustration if CSR is not helping with cancellation
        csr_lower = csr_message.lower()
        
        if any(word in csr_lower for word in ["לא יכול", "cannot", "אי אפשר", "impossible"]):
            # CSR refusing or making it difficult
            self.state.frustration_level = min(5, self.state.frustration_level + 2)
        
        elif any(word in csr_lower for word in ["הצעה", "הנחה", "offer", "discount"]):
            # CSR making offers instead of canceling
            self.state.frustration_level = min(5, self.state.frustration_level + 1)
        
        elif any(word in csr_lower for word in ["בוטל", "canceled", "סגור", "closed"]):
            # Cancellation confirmed
            self.state.satisfied = True
            self.state.frustration_level = 1
        
        # Natural frustration increase over time
        if self.state.attempts > 3 and not self.state.satisfied:
            self.state.frustration_level = min(5, self.state.frustration_level + 1)
    
    def _get_speech_speed(self) -> float:
        """Get speech speed based on frustration level."""
        speed_map = {
            1: 0.9,   # Calm, slightly slower
            2: 1.0,   # Normal speed
            3: 1.1,   # Slightly faster
            4: 1.2,   # Faster, more urgent
            5: 1.3    # Very fast, frustrated
        }
        return speed_map.get(self.state.frustration_level, 1.0)
    
    def reset_state(self):
        """Reset client state for new conversation."""
        self.state = ClientState()
        self.turn_counter = 0
        logger.info("Client state reset")
    
    def create_task(self, csr_message: str = "", context: str = "") -> Task:
        """
        Create a CrewAI task for client response generation.
        
        Args:
            csr_message: Message from CSR
            context: Additional context
            
        Returns:
            CrewAI Task object
        """
        return Task(
            description=f"""
            Generate a natural Hebrew response as a customer trying to cancel TV subscription:
            
            CSR Message: {csr_message}
            Context: {context}
            Current State: {self.state.dict()}
            Turn: {self.turn_counter + 1}
            Personality: {self.personality}
            
            Requirements:
            1. Respond naturally in Hebrew
            2. Stay in character as customer wanting to cancel
            3. Adjust tone based on frustration level
            4. Use appropriate Hebrew phrases for the situation
            5. Generate audio using TTS
            """,
            agent=self.agent,
            expected_output="ClientResponse object with Hebrew text, nikud, and audio file"
        )
    
    def execute_task(self, task: Task) -> ClientResponse:
        """Execute client response generation task."""
        # Parse task parameters
        lines = task.description.strip().split('\n')
        csr_message = ""
        context = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("CSR Message:"):
                csr_message = line.split("CSR Message:", 1)[1].strip()
            elif line.startswith("Context:"):
                context = line.split("Context:", 1)[1].strip()
        
        return self.generate_response(csr_message, context)
    
    def get_stats(self) -> Dict:
        """Get client agent statistics."""
        return {
            "personality": self.personality,
            "current_state": self.state.dict(),
            "turn_counter": self.turn_counter,
            "phrases_loaded": len(self.phrases),
            "nikud_agent_available": self.nikud_agent.is_service_available(),
            "tts_agent_available": self.tts_agent.is_service_available()
        }
