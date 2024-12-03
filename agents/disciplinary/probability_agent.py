from typing import Any, Dict, List, Set
import re
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class ProbabilityStructure:
    """Data class for storing probabilistic structures."""
    name: str
    properties: Set[str]
    measures: List[str]
    convergence: List[str]
    processes: List[str]
    theorems: List[str]

class ProbabilityTheoryAgent(BaseAgent):
    """Agent specialized in reviewing Probability Theory papers."""
    
    def __init__(self, name: str = "Probability Theory Reviewer", model_name: str = "allenai/mathbert-base-cased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.structures = self._initialize_structures()
        self.measure_theory = {
            "basic": r"(?i)(measure|measurable|sigma-algebra|borel)",
            "probability": r"(?i)(probability measure|distribution|density|random variable)",
            "integration": r"(?i)(lebesgue integral|expectation|moment|characteristic function)",
            "absolute": r"(?i)(absolute continuity|radon-nikodym|singular|mutually singular)"
        }
        self.convergence = {
            "modes": r"(?i)(almost surely|in probability|L\^p|distribution)",
            "limit": r"(?i)(law of large numbers|central limit|ergodic|martingale)",
            "rates": r"(?i)(rate of convergence|large deviation|moderate deviation)",
            "inequalities": r"(?i)(markov inequality|chebyshev|hoeffding|bernstein)"
        }
        self.stochastic_processes = {
            "markov": r"(?i)(markov chain|transition|stationary|ergodic)",
            "martingales": r"(?i)(martingale|stopping time|filtration|predictable)",
            "gaussian": r"(?i)(gaussian process|brownian motion|wiener|gaussian field)",
            "levy": r"(?i)(lévy process|jump process|subordinator|stable process)"
        }
        self.applications = {
            "statistics": r"(?i)(statistical inference|estimation|hypothesis test|confidence)",
            "finance": r"(?i)(financial mathematics|option pricing|black-scholes|portfolio)",
            "physics": r"(?i)(statistical mechanics|quantum probability|diffusion|entropy)",
            "computing": r"(?i)(monte carlo|mcmc|simulation|random algorithm)"
        }
    
    def _initialize_structures(self) -> Dict[str, ProbabilityStructure]:
        """Initialize common probabilistic structures."""
        return {
            "measure": ProbabilityStructure(
                name="Measure Theory",
                properties={"countably additive", "complete", "sigma-finite"},
                measures=["Probability measure", "Lebesgue measure", "Haar measure"],
                convergence=["Almost everywhere", "In measure", "Dominated"],
                processes=["Random variables", "Stochastic processes"],
                theorems=["Radon-Nikodym", "Fubini", "Monotone convergence"]
            ),
            "limit": ProbabilityStructure(
                name="Limit Theorems",
                properties={"independence", "identical distribution", "stationarity"},
                measures=["Probability", "Distribution", "Characteristic function"],
                convergence=["Almost sure", "In probability", "In distribution"],
                processes=["Sums", "Maxima", "Empirical processes"],
                theorems=["Law of large numbers", "Central limit", "Berry-Esseen"]
            ),
            "process": ProbabilityStructure(
                name="Stochastic Process",
                properties={"adaptedness", "cadlag", "continuous"},
                measures=["Path measure", "Wiener measure", "Point process"],
                convergence=["Weak convergence", "Functional CLT", "Invariance"],
                processes=["Markov", "Martingale", "Gaussian"],
                theorems=["Doob's martingale", "Donsker", "Kolmogorov extension"]
            ),
            "statistical": ProbabilityStructure(
                name="Statistical Theory",
                properties={"unbiased", "consistent", "efficient"},
                measures=["Sampling distribution", "Prior", "Posterior"],
                convergence=["Asymptotic normality", "Consistency", "Efficiency"],
                processes=["Estimators", "Test statistics", "Empirical"],
                theorems=["Cramér-Rao", "Neyman-Pearson", "Glivenko-Cantelli"]
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the probability theory model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.structures)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _check_measure_theory(self, text: str) -> Dict[str, Any]:
        """Analyze usage of measure theory concepts."""
        results = {
            "concepts_used": [],
            "missing_properties": [],
            "incorrect_usage": []
        }
        
        for concept_type, pattern in self.measure_theory.items():
            if re.search(pattern, text):
                results["concepts_used"].append(concept_type)
                
                # Check for proper properties
                if concept_type == "basic":
                    if not re.search(r"(?i)(sigma-additivity|completion|generator)", text):
                        results["missing_properties"].append("Basic measure properties not verified")
                
                elif concept_type == "integration":
                    if not re.search(r"(?i)(measurable|integrability|convergence)", text):
                        results["missing_properties"].append("Integration theory not properly developed")
        
        return results
    
    def _analyze_convergence(self, text: str) -> Dict[str, List[str]]:
        """Analyze convergence concepts and properties."""
        convergence = {
            "identified": [],
            "properties_verified": [],
            "issues": []
        }
        
        for conv_type, pattern in self.convergence.items():
            if re.search(pattern, text):
                convergence["identified"].append(conv_type)
                
                # Check for necessary properties
                if conv_type == "modes":
                    if not re.search(r"(?i)(almost sure|probability|measure)", text):
                        convergence["issues"].append("Convergence mode not properly specified")
                
                elif conv_type == "limit":
                    if not re.search(r"(?i)(theorem|proof|verification)", text):
                        convergence["issues"].append("Limit theorem not properly established")
        
        return convergence
    
    def _check_processes(self, text: str) -> Dict[str, Any]:
        """Analyze stochastic process properties."""
        results = {
            "processes": [],
            "properties": [],
            "missing_arguments": []
        }
        
        for proc_type, pattern in self.stochastic_processes.items():
            if re.search(pattern, text):
                results["processes"].append(proc_type)
                
                # Check for specific requirements
                if proc_type == "markov":
                    if not re.search(r"(?i)(transition|semigroup|generator)", text):
                        results["missing_arguments"].append("Markov property not properly verified")
                
                elif proc_type == "martingales":
                    if not re.search(r"(?i)(filtration|adapted|integrable)", text):
                        results["missing_arguments"].append("Martingale properties not established")
        
        return results
    
    def _analyze_applications(self, text: str) -> Dict[str, Any]:
        """Analyze applications of probability theory."""
        results = {
            "areas": [],
            "methods": [],
            "techniques": [],
            "issues": []
        }
        
        for app_type, pattern in self.applications.items():
            if re.search(pattern, text):
                results["areas"].append(app_type)
                
                # Check specific aspects
                if app_type == "statistics":
                    if not re.search(r"(?i)(estimator|test|asymptotic|efficiency)", text):
                        results["issues"].append("Statistical framework not properly developed")
                
                elif app_type == "finance":
                    if not re.search(r"(?i)(martingale|arbitrage|hedging|risk)", text):
                        results["issues"].append("Financial mathematics framework incomplete")
        
        return results
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform probability theory specific review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze different aspects
        measure = self._check_measure_theory(content)
        convergence = self._analyze_convergence(content)
        processes = self._check_processes(content)
        applications = self._analyze_applications(content)
        
        # Generate comments and suggestions
        comments = []
        suggestions = []
        
        # Measure theory analysis
        if measure["concepts_used"]:
            comments.append(f"Measure theory concepts: {', '.join(measure['concepts_used'])}")
        if measure["missing_properties"]:
            comments.extend(measure["missing_properties"])
            suggestions.append("Verify measure-theoretic properties")
        
        # Convergence analysis
        if convergence["identified"]:
            comments.append(f"Convergence concepts: {', '.join(convergence['identified'])}")
        if convergence["issues"]:
            comments.extend(convergence["issues"])
            suggestions.append("Complete convergence arguments")
        
        # Process analysis
        if processes["processes"]:
            comments.append(f"Stochastic processes: {', '.join(processes['processes'])}")
        if processes["missing_arguments"]:
            comments.extend(processes["missing_arguments"])
            suggestions.append("Verify process properties")
        
        # Application analysis
        if applications["areas"]:
            comments.append(f"Applications: {', '.join(applications['areas'])}")
        if applications["issues"]:
            comments.extend(applications["issues"])
            suggestions.append("Develop application theory fully")
        
        # Calculate score
        score = 1.0
        score -= 0.1 * len(measure["missing_properties"])
        score -= 0.1 * len(convergence["issues"])
        score -= 0.1 * len(processes["missing_arguments"])
        score -= 0.1 * len(applications["issues"])
        score = max(0.0, score)
        
        # Calculate confidence
        confidence = 0.9 if (
            measure["concepts_used"] or
            convergence["identified"] or
            processes["processes"] or
            applications["areas"]
        ) else 0.7
        
        return ReviewResult(
            score=score,
            comments=comments,
            suggestions=suggestions,
            confidence=confidence,
            metadata={
                "measure_theory": measure,
                "convergence": convergence,
                "processes": processes,
                "applications": applications
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing probabilistic insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "probability_score": review_result.score,
            "measure_concepts": review_result.metadata["measure_theory"]["concepts_used"],
            "convergence_types": review_result.metadata["convergence"]["identified"],
            "process_types": review_result.metadata["processes"]["processes"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Probability Theory paper review",
            "Measure theory verification",
            "Convergence analysis",
            "Stochastic process assessment",
            "Application verification",
            "Limit theorem computations",
            "Mathematical rigor assessment",
            "Cross-field applications"
        ] 