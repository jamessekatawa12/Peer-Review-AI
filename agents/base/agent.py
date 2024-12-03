from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    DISCIPLINARY = "disciplinary"
    INTERDISCIPLINARY = "interdisciplinary"
    ETHICS = "ethics"

@dataclass
class ReviewResult:
    """Data class for storing review results."""
    score: float
    comments: List[str]
    suggestions: List[str]
    confidence: float
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    """Abstract base class for all peer review agents."""
    
    def __init__(self, name: str, agent_type: AgentType, model_path: Optional[str] = None):
        self.name = name
        self.agent_type = agent_type
        self.model_path = model_path
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent and load necessary models."""
        pass
    
    @abstractmethod
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform review on the given content."""
        pass
    
    @abstractmethod
    async def collaborate(self, other_agent: 'BaseAgent', shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with another agent."""
        pass
    
    async def validate(self, content: str) -> bool:
        """Validate if the content is suitable for review by this agent."""
        return True
    
    def get_capabilities(self) -> List[str]:
        """Return a list of agent capabilities."""
        return []
    
    def __str__(self) -> str:
        return f"{self.agent_type.value.capitalize()} Agent: {self.name}" 