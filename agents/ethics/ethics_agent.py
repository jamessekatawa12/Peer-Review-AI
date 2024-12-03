from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from ..base.agent import BaseAgent, AgentType, ReviewResult

class EthicsReviewAgent(BaseAgent):
    """Agent specialized in evaluating ethical considerations in research."""
    
    def __init__(self, name: str = "Ethics Reviewer", model_name: str = "microsoft/deberta-v3-base"):
        super().__init__(name, AgentType.ETHICS)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.ethical_categories = [
            "human_subjects",
            "privacy",
            "bias",
            "environmental_impact",
            "dual_use",
            "conflict_of_interest"
        ]
        
    async def initialize(self) -> None:
        """Initialize the ethics analysis model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.ethical_categories)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _analyze_ethical_concerns(self, text: str) -> Dict[str, float]:
        """Analyze text for various ethical considerations."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze().numpy()
        
        return dict(zip(self.ethical_categories, scores))
    
    def _generate_recommendations(self, concerns: Dict[str, float]) -> List[str]:
        """Generate specific recommendations based on identified concerns."""
        recommendations = []
        
        if concerns["human_subjects"] > 0.5:
            recommendations.append(
                "Ensure proper IRB approval and informed consent procedures are in place."
            )
        
        if concerns["privacy"] > 0.5:
            recommendations.append(
                "Review data anonymization and protection measures. Consider GDPR and other privacy regulations."
            )
        
        if concerns["bias"] > 0.5:
            recommendations.append(
                "Address potential biases in methodology and data collection. Consider impacts on underrepresented groups."
            )
        
        if concerns["environmental_impact"] > 0.5:
            recommendations.append(
                "Evaluate and document the environmental impact of the research methods and outcomes."
            )
        
        if concerns["dual_use"] > 0.5:
            recommendations.append(
                "Consider potential dual-use implications and include appropriate safeguards."
            )
        
        if concerns["conflict_of_interest"] > 0.5:
            recommendations.append(
                "Clearly disclose any potential conflicts of interest or funding sources."
            )
        
        return recommendations
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform ethical review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze ethical concerns
        concerns = self._analyze_ethical_concerns(content)
        
        # Generate overall ethics score (inverse of average concern level)
        ethics_score = 1.0 - np.mean(list(concerns.values()))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(concerns)
        
        # Prepare comments based on significant concerns
        comments = [
            f"Ethical concern identified: {category.replace('_', ' ').title()}"
            for category, score in concerns.items()
            if score > 0.5
        ]
        
        if not comments:
            comments = ["No major ethical concerns identified."]
        
        return ReviewResult(
            score=ethics_score,
            comments=comments,
            suggestions=recommendations,
            confidence=0.85,
            metadata={
                "ethical_concerns": concerns,
                "num_concerns": len([s for s in concerns.values() if s > 0.5])
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing ethical insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "ethical_score": review_result.score,
            "ethical_concerns": review_result.metadata["ethical_concerns"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Ethical analysis of research manuscripts",
            "Human subjects research evaluation",
            "Privacy and data protection assessment",
            "Bias detection and analysis",
            "Environmental impact evaluation",
            "Dual-use implications assessment",
            "Conflict of interest detection"
        ] 