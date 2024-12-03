from typing import Any, Dict, List, Set
import re
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class FunctionalStructure:
    """Data class for storing functional analytic structures."""
    name: str
    properties: Set[str]
    topology: List[str]
    operators: List[str]
    duality: List[str]
    theorems: List[str]

class FunctionalAnalysisAgent(BaseAgent):
    """Agent specialized in reviewing Functional Analysis papers."""
    
    def __init__(self, name: str = "Functional Analysis Reviewer", model_name: str = "allenai/mathbert-base-cased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.structures = self._initialize_structures()
        self.spaces = {
            "normed": r"(?i)(normed space|banach space|complete|norm)",
            "hilbert": r"(?i)(hilbert space|inner product|orthogonal|self-adjoint)",
            "locally_convex": r"(?i)(locally convex|seminorm|fréchet|nuclear)",
            "function": r"(?i)(L\^p|sobolev|hölder|hardy)"
        }
        self.operators = {
            "bounded": r"(?i)(bounded operator|continuous|operator norm|adjoint)",
            "unbounded": r"(?i)(unbounded|closed operator|domain|spectrum)",
            "compact": r"(?i)(compact operator|completely continuous|nuclear|trace class)",
            "spectral": r"(?i)(spectral theory|eigenvalue|resolvent|functional calculus)"
        }
        self.topology = {
            "weak": r"(?i)(weak topology|weak-\*|weak convergence|duality)",
            "strong": r"(?i)(strong topology|norm topology|uniform|bounded)",
            "operator": r"(?i)(operator topology|strong operator|weak operator)",
            "convergence": r"(?i)(convergence|nets|filters|compactness)"
        }
        self.applications = {
            "differential": r"(?i)(differential operator|elliptic|evolution equation|semigroup)",
            "integral": r"(?i)(integral operator|fredholm|volterra|kernel)",
            "quantum": r"(?i)(quantum mechanics|observable|state|C\*-algebra)",
            "numerical": r"(?i)(approximation|discretization|stability|error estimate)"
        }
    
    def _initialize_structures(self) -> Dict[str, FunctionalStructure]:
        """Initialize common functional analytic structures."""
        return {
            "banach": FunctionalStructure(
                name="Banach Space",
                properties={"complete", "normed", "separable"},
                topology=["Norm topology", "Weak topology", "Weak* topology"],
                operators=["Bounded", "Compact", "Fredholm"],
                duality=["Dual space", "Bidual", "Reflexivity"],
                theorems=["Hahn-Banach", "Open mapping", "Closed graph"]
            ),
            "hilbert": FunctionalStructure(
                name="Hilbert Space",
                properties={"inner product", "complete", "separable"},
                topology=["Strong topology", "Weak topology"],
                operators=["Self-adjoint", "Unitary", "Normal"],
                duality=["Riesz representation", "Orthogonal complement"],
                theorems=["Projection theorem", "Spectral theorem", "Lax-Milgram"]
            ),
            "operator": FunctionalStructure(
                name="Operator Theory",
                properties={"bounded", "closed", "densely defined"},
                topology=["Operator norm", "Strong operator", "Weak operator"],
                operators=["Compact", "Self-adjoint", "Normal"],
                duality=["Adjoint", "Spectrum", "Resolvent"],
                theorems=["Spectral theorem", "Fredholm alternative", "Gelfand-Naimark"]
            ),
            "locally_convex": FunctionalStructure(
                name="Locally Convex Space",
                properties={"topological", "hausdorff", "barrelled"},
                topology=["Seminorm topology", "Weak topology", "Mackey topology"],
                operators=["Continuous", "Nuclear", "Schwartz"],
                duality=["Strong dual", "Polar sets", "Equicontinuous"],
                theorems=["Banach-Alaoglu", "Krein-Milman", "Mackey-Arens"]
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the functional analysis model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.structures)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _check_spaces(self, text: str) -> Dict[str, Any]:
        """Analyze usage of functional spaces."""
        results = {
            "spaces_used": [],
            "missing_properties": [],
            "incorrect_usage": []
        }
        
        for space_type, pattern in self.spaces.items():
            if re.search(pattern, text):
                results["spaces_used"].append(space_type)
                
                # Check for proper properties
                if space_type == "normed":
                    if not re.search(r"(?i)(complete|cauchy|bounded|closed)", text):
                        results["missing_properties"].append("Completeness/boundedness not verified")
                
                elif space_type == "hilbert":
                    if not re.search(r"(?i)(inner product|orthogonal|complete)", text):
                        results["missing_properties"].append("Inner product space properties not established")
        
        return results
    
    def _analyze_operators(self, text: str) -> Dict[str, List[str]]:
        """Analyze operator theory concepts and properties."""
        operators = {
            "identified": [],
            "properties_verified": [],
            "issues": []
        }
        
        for op_type, pattern in self.operators.items():
            if re.search(pattern, text):
                operators["identified"].append(op_type)
                
                # Check for necessary properties
                if op_type == "bounded":
                    if not re.search(r"(?i)(continuous|norm|domain|range)", text):
                        operators["issues"].append("Operator boundedness not properly verified")
                
                elif op_type == "spectral":
                    if not re.search(r"(?i)(spectrum|resolvent|eigenvalue)", text):
                        operators["issues"].append("Spectral properties not properly established")
        
        return operators
    
    def _check_topology(self, text: str) -> Dict[str, Any]:
        """Analyze topological aspects."""
        results = {
            "topologies": [],
            "convergence": [],
            "missing_arguments": []
        }
        
        for top_type, pattern in self.topology.items():
            if re.search(pattern, text):
                results["topologies"].append(top_type)
                
                # Check for specific requirements
                if top_type == "weak":
                    if not re.search(r"(?i)(continuous functionals|duality|separation)", text):
                        results["missing_arguments"].append("Weak topology properties not verified")
                
                elif top_type == "operator":
                    if not re.search(r"(?i)(convergence|bounded|uniform)", text):
                        results["missing_arguments"].append("Operator topology not properly defined")
        
        return results
    
    def _analyze_applications(self, text: str) -> Dict[str, Any]:
        """Analyze applications of functional analysis."""
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
                if app_type == "differential":
                    if not re.search(r"(?i)(boundary|initial|solution|regularity)", text):
                        results["issues"].append("Differential operator theory incomplete")
                
                elif app_type == "quantum":
                    if not re.search(r"(?i)(hermitian|unitary|state|observable)", text):
                        results["issues"].append("Quantum mechanical framework not properly established")
        
        return results
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform functional analysis specific review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze different aspects
        spaces = self._check_spaces(content)
        operators = self._analyze_operators(content)
        topology = self._check_topology(content)
        applications = self._analyze_applications(content)
        
        # Generate comments and suggestions
        comments = []
        suggestions = []
        
        # Space analysis
        if spaces["spaces_used"]:
            comments.append(f"Functional spaces used: {', '.join(spaces['spaces_used'])}")
        if spaces["missing_properties"]:
            comments.extend(spaces["missing_properties"])
            suggestions.append("Verify all space properties")
        
        # Operator analysis
        if operators["identified"]:
            comments.append(f"Operator types: {', '.join(operators['identified'])}")
        if operators["issues"]:
            comments.extend(operators["issues"])
            suggestions.append("Complete operator theory arguments")
        
        # Topology analysis
        if topology["topologies"]:
            comments.append(f"Topological aspects: {', '.join(topology['topologies'])}")
        if topology["missing_arguments"]:
            comments.extend(topology["missing_arguments"])
            suggestions.append("Verify topological properties")
        
        # Application analysis
        if applications["areas"]:
            comments.append(f"Applications: {', '.join(applications['areas'])}")
        if applications["issues"]:
            comments.extend(applications["issues"])
            suggestions.append("Develop application theory fully")
        
        # Calculate score
        score = 1.0
        score -= 0.1 * len(spaces["missing_properties"])
        score -= 0.1 * len(operators["issues"])
        score -= 0.1 * len(topology["missing_arguments"])
        score -= 0.1 * len(applications["issues"])
        score = max(0.0, score)
        
        # Calculate confidence
        confidence = 0.9 if (
            spaces["spaces_used"] or
            operators["identified"] or
            topology["topologies"] or
            applications["areas"]
        ) else 0.7
        
        return ReviewResult(
            score=score,
            comments=comments,
            suggestions=suggestions,
            confidence=confidence,
            metadata={
                "spaces": spaces,
                "operators": operators,
                "topology": topology,
                "applications": applications
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing functional analytic insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "functional_analysis_score": review_result.score,
            "spaces_used": review_result.metadata["spaces"]["spaces_used"],
            "operator_types": review_result.metadata["operators"]["identified"],
            "topological_aspects": review_result.metadata["topology"]["topologies"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Functional Analysis paper review",
            "Space theory verification",
            "Operator theory analysis",
            "Topological assessment",
            "Application verification",
            "Spectral theory computations",
            "Mathematical rigor assessment",
            "Cross-field applications"
        ] 