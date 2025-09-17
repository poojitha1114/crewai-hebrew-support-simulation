"""
CSRAgent: Customer Service Representative agent for handling Hebrew customer interactions.
"""

import logging
from typing import Dict, Optional
from pydantic import BaseModel
from crewai import Agent

logger = logging.getLogger(__name__)

class CSRState(BaseModel):
    """Current state of the CSR conversation."""
    cancellation_approved: bool = False
    offers_made: int = 0
    escalation_level: int = 0

class CSRResponse(BaseModel):
    """Response from CSR agent."""
    text: str
    action: str
    tokens_used: int = 0
    state: CSRState
    success: bool = True
    error_message: Optional[str] = None

class CSRAgent:
    """Customer Service Representative agent for Hebrew conversations."""
    
    def __init__(self, model_config: Dict):
        """Initialize CSRAgent."""
        self.model_config = model_config
        self.state = CSRState()
        self.conversation_history = []
        logger.info("CSRAgent initialized")
    
    def generate_response(self, client_message: str, context: str = "") -> CSRResponse:
        """Generate CSR response to client message."""
        try:
            # Simple response logic for demo
            responses = [
                "שלום! איך אני יכול לעזור לך היום?",
                "אני מבין את בקשתך. בואו נבדוק את האפשרויות שלך.",
                "יש לנו הצעות מיוחדות שעשויות לעניין אותך.",
                "אני מצטער על אי הנוחות. בואו נפתור את זה יחד.",
                "תודה על סבלנותך. אני אסדר את זה עבורך עכשיו."
            ]
            
            # Select response based on conversation length
            response_idx = min(len(self.conversation_history), len(responses) - 1)
            response_text = responses[response_idx]
            
            # Update conversation history
            self.conversation_history.append({
                "client": client_message,
                "csr": response_text
            })
            
            # Determine action
            action = "respond"
            if len(self.conversation_history) > 3:
                action = "resolve"
                self.state.cancellation_approved = True
            
            return CSRResponse(
                text=response_text,
                action=action,
                tokens_used=50,  # Estimated
                state=self.state,
                success=True
            )
            
        except Exception as e:
            logger.error(f"CSR response generation failed: {e}")
            return CSRResponse(
                text="",
                action="error",
                state=self.state,
                success=False,
                error_message=str(e)
            )
    
    def reset_state(self):
        """Reset CSR state for new conversation."""
        self.state = CSRState()
        self.conversation_history = []
    
    def get_stats(self) -> Dict:
        """Get CSR statistics."""
        return {
            "total_responses": len(self.conversation_history),
            "offers_made": self.state.offers_made,
            "escalation_level": self.state.escalation_level
        }
