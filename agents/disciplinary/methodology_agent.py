from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import re
from ..base.agent import BaseAgent, AgentType, ReviewResult

class MethodologyAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing research methodology."""
    
    def __init__(self, name: str = "Methodology Analyzer", model_name: str = "allenai/scibert_scivocab_uncased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.nlp = None
        self.methodology_aspects = [
            "research_design",
            "data_collection",
            "sampling",
            "analysis_methods",
            "validity",
            "reliability",
            "reproducibility"
        ]
    
    async def initialize(self) -> None:
        """Initialize the methodology analysis models."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.methodology_aspects)
            )
            self.nlp = spacy.load("en_core_web_sm")
            self.model.eval()
            self.is_initialized = True
    
    def _extract_methodology_section(self, content: str) -> str:
        """Extract the methodology section from the manuscript."""
        # Simple heuristic to find methodology section
        sections = re.split(r'\n(?=[A-Z][a-z]+ *\n|[0-9]+\. [A-Z])', content)
        methodology_section = ""
        
        for section in sections:
            if any(keyword in section.lower() for keyword in ["method", "methodology", "experimental setup", "procedure"]):
                methodology_section += section + "\n"
        
        return methodology_section if methodology_section else content
    
    def _analyze_methodology(self, text: str) -> Dict[str, float]:
        """Analyze various aspects of the research methodology."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze().numpy()
        
        return dict(zip(self.methodology_aspects, scores))
    
    def _check_statistical_rigor(self, text: str) -> List[str]:
        """Check for statistical analysis and rigor."""
        doc = self.nlp(text)
        issues = []
        
        # Check for statistical terms and potential issues
        statistical_terms = ["p-value", "significance", "correlation", "regression", "anova", "t-test"]
        found_stats = set()
        
        for term in statistical_terms:
            if term in text.lower():
                found_stats.add(term)
        
        if not found_stats:
            issues.append("No statistical analysis methods mentioned")
        else:
            if "p-value" in found_stats and not re.search(r'p\s*[<>=]\s*0\.\d+', text):
                issues.append("P-values mentioned but not properly reported")
            
            if "significance" in found_stats and not re.search(r'statistical(ly)?\s+significant', text, re.I):
                issues.append("Significance mentioned but not properly defined")
        
        return issues
    
    def _generate_recommendations(self, scores: Dict[str, float], stats_issues: List[str]) -> List[str]:
        """Generate recommendations based on methodology analysis."""
        recommendations = []
        
        if scores["research_design"] < 0.7:
            recommendations.append(
                "Strengthen research design description with clear objectives and hypotheses"
            )
        
        if scores["data_collection"] < 0.7:
            recommendations.append(
                "Provide more detailed information about data collection procedures"
            )
        
        if scores["sampling"] < 0.7:
            recommendations.append(
                "Include more information about sampling methods and sample size justification"
            )
        
        if scores["analysis_methods"] < 0.7:
            recommendations.append(
                "Enhance description of analysis methods and statistical procedures"
            )
        
        if scores["validity"] < 0.7:
            recommendations.append(
                "Address internal and external validity concerns"
            )
        
        if scores["reliability"] < 0.7:
            recommendations.append(
                "Include reliability measures and assessment procedures"
            )
        
        if scores["reproducibility"] < 0.7:
            recommendations.append(
                "Provide more detailed information to ensure reproducibility"
            )
        
        recommendations.extend(stats_issues)
        return recommendations
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform methodology review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Extract and analyze methodology section
        methodology_section = self._extract_methodology_section(content)
        methodology_scores = self._analyze_methodology(methodology_section)
        statistical_issues = self._check_statistical_rigor(methodology_section)
        
        # Calculate overall methodology score
        methodology_score = sum(methodology_scores.values()) / len(methodology_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(methodology_scores, statistical_issues)
        
        # Prepare comments
        comments = [
            f"Methodology aspect needs improvement: {aspect.replace('_', ' ').title()}"
            for aspect, score in methodology_scores.items()
            if score < 0.7
        ]
        
        if not comments:
            comments = ["Methodology appears to be well-described and appropriate."]
        
        return ReviewResult(
            score=methodology_score,
            comments=comments,
            suggestions=recommendations,
            confidence=0.85,
            metadata={
                "methodology_scores": methodology_scores,
                "statistical_issues": statistical_issues,
                "section_length": len(methodology_section.split())
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing methodology insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "methodology_score": review_result.score,
            "methodology_aspects": review_result.metadata["methodology_scores"],
            "statistical_issues": review_result.metadata["statistical_issues"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Research methodology analysis",
            "Statistical rigor assessment",
            "Research design evaluation",
            "Data collection methods analysis",
            "Sampling methodology review",
            "Validity and reliability assessment",
            "Reproducibility evaluation"
        ] 