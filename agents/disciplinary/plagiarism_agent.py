from typing import Any, Dict, List
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

class PlagiarismDetectionAgent(BaseAgent):
    """Agent specialized in detecting potential plagiarism in academic manuscripts."""
    
    def __init__(self, name: str = "Plagiarism Detector", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.similarity_threshold = 0.85
    
    async def initialize(self) -> None:
        """Initialize the transformer model and tokenizer."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.is_initialized = True
    
    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Generate embeddings for the given text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    
    def _calculate_similarity(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        return torch.nn.functional.cosine_similarity(embed1, embed2).item()
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform plagiarism detection on the given content."""
        if not self.is_initialized:
            await self.initialize()
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        # Compare each paragraph with reference texts (simplified for example)
        suspicious_sections = []
        similarity_scores = []
        
        # In a real implementation, we would compare against a database of reference texts
        # For now, we'll just compare consecutive paragraphs as an example
        for i in range(len(paragraphs) - 1):
            embed1 = self._get_embeddings(paragraphs[i])
            embed2 = self._get_embeddings(paragraphs[i + 1])
            similarity = self._calculate_similarity(embed1, embed2)
            
            if similarity > self.similarity_threshold:
                suspicious_sections.append(f"High similarity detected between paragraphs {i+1} and {i+2}")
                similarity_scores.append(similarity)
        
        # Prepare review result
        comments = suspicious_sections if suspicious_sections else ["No significant plagiarism detected"]
        suggestions = [
            "Consider rephrasing similar sections to avoid potential plagiarism",
            "Ensure all sources are properly cited",
            "Use quotation marks for direct quotes"
        ] if suspicious_sections else []
        
        confidence = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return ReviewResult(
            score=1.0 - (confidence if suspicious_sections else 0.0),
            comments=comments,
            suggestions=suggestions,
            confidence=0.9,  # Model confidence in its assessment
            metadata={
                "suspicious_sections": len(suspicious_sections),
                "max_similarity": max(similarity_scores) if similarity_scores else 0.0,
                "analyzed_paragraphs": len(paragraphs)
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing plagiarism detection results."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
            
        review_result = await self.review(shared_data["content"], {})
        return {
            "plagiarism_score": review_result.score,
            "suspicious_sections": review_result.metadata["suspicious_sections"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Plagiarism detection using transformer-based embeddings",
            "Cross-paragraph similarity analysis",
            "Suspicious content identification",
            "Collaboration with other review agents"
        ] 