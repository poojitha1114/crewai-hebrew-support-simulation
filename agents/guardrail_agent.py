"""
GuardrailAgent: Provides safety and content filtering for conversations.
"""

import logging
from typing import Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class GuardrailConfig(BaseModel):
    """Guardrail configuration."""
    max_retries: int = 3
    min_stt_confidence: float = 0.6
    max_conversation_turns: int = 15

class GuardrailViolation(BaseModel):
    """Guardrail violation details."""
    type: str
    severity: str
    message: str

class GuardrailResult(BaseModel):
    """Result of guardrail check."""
    allowed: bool = True
    violations: List[GuardrailViolation] = []
    modified_content: Optional[str] = None
    conversation_should_end: bool = False

class GuardrailAgent:
    """Agent for content filtering and conversation safety."""
    
    def __init__(self, config: GuardrailConfig):
        """Initialize GuardrailAgent."""
        self.config = config
        self.violation_count = 0
        logger.info("GuardrailAgent initialized")
    
    def check_content(self, content: str, content_type: str, 
                     speaker: str, turn_number: int) -> GuardrailResult:
        """Check content against guardrails."""
        violations = []
        conversation_should_end = False
        
        # Check turn limit
        if content_type == "turn_check" and turn_number >= self.config.max_conversation_turns:
            conversation_should_end = True
            violations.append(GuardrailViolation(
                type="turn_limit",
                severity="info",
                message="Maximum conversation turns reached"
            ))
        
        # Basic content checks (placeholder)
        if len(content) > 1000:
            violations.append(GuardrailViolation(
                type="length",
                severity="warning", 
                message="Content too long"
            ))
        
        return GuardrailResult(
            allowed=len(violations) == 0 or all(v.severity != "error" for v in violations),
            violations=violations,
            conversation_should_end=conversation_should_end
        )
    
    def get_stats(self) -> Dict:
        """Get guardrail statistics."""
        return {
            "violation_count": self.violation_count
        }
