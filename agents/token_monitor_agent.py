"""
TokenMonitorAgent: Monitors and manages token usage during conversations.
"""

import logging
from typing import Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TokenBudget(BaseModel):
    """Token budget configuration."""
    max_tokens_per_conversation: int = 10000
    warning_threshold: float = 0.8
    summarization_threshold: float = 0.9

class TokenUsage(BaseModel):
    """Current token usage statistics."""
    total_tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    summarizations_triggered: int = 0

class TokenStatus(BaseModel):
    """Token monitoring status."""
    current_usage: TokenUsage
    budget_remaining: int
    action_required: str = "none"  # "none", "warning", "summarize", "stop"

class TokenMonitorAgent:
    """Agent for monitoring and managing token usage."""
    
    def __init__(self, budget_config: TokenBudget):
        """Initialize TokenMonitorAgent."""
        self.budget = budget_config
        self.usage = TokenUsage()
        logger.info(f"TokenMonitorAgent initialized with budget: {budget_config.max_tokens_per_conversation}")
    
    def record_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> TokenStatus:
        """Record token usage and check against budget."""
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.total_tokens_used += total_tokens
        
        budget_remaining = self.budget.max_tokens_per_conversation - self.usage.total_tokens_used
        usage_ratio = self.usage.total_tokens_used / self.budget.max_tokens_per_conversation
        
        # Determine action required
        action_required = "none"
        if usage_ratio >= self.budget.summarization_threshold:
            action_required = "summarize"
        elif usage_ratio >= self.budget.warning_threshold:
            action_required = "warning"
        
        return TokenStatus(
            current_usage=self.usage,
            budget_remaining=budget_remaining,
            action_required=action_required
        )
    
    def trigger_summarization(self):
        """Trigger conversation summarization."""
        self.usage.summarizations_triggered += 1
        # Reset token count after summarization
        self.usage.total_tokens_used = self.usage.total_tokens_used // 2
        logger.info("Conversation summarization triggered")
    
    def reset_conversation_usage(self):
        """Reset usage for new conversation."""
        self.usage = TokenUsage()
    
    def get_stats(self) -> Dict:
        """Get token usage statistics."""
        return {
            "current_usage": self.usage.dict(),
            "budget": self.budget.dict()
        }
