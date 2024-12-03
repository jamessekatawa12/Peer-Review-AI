from typing import Any, Dict, List, Set
import re
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class ComplexStructure:
    """Data class for storing complex analytic structures."""
    name: str
    properties: Set[str]
    local_theory: List[str]
    global_theory: List[str]
    applications: List[str]
    theorems: List[str]

class ComplexAnalysisAgent(BaseAgent):
    """Agent specialized in reviewing Complex Analysis papers."""
    
    def __init__(self, name: str = "Complex Analysis Reviewer", model_name: str = "allenai/mathbert-base-cased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.structures = self._initialize_structures()
        self.holomorphic = {
            "basic": r"(?i)(holomorphic|analytic|meromorphic|entire)",
            "derivatives": r"(?i)(cauchy-riemann|complex derivative|conformal|biholomorphic)",
            "singularities": r"(?i)(pole|essential|removable|branch point)",
            "zeros": r"(?i)(zero|multiplicity|order|factorization)"
        }
        self.integration = {
            "contour": r"(?i)(contour integral|path integral|closed path|winding number)",
            "residues": r"(?i)(residue|principal part|laurent series|pole order)",
            "theorems": r"(?i)(cauchy integral|morera|goursat|argument principle)",
            "applications": r"(?i)(evaluation|real integral|fourier transform|laplace)"
        }
        self.riemann_surfaces = {
            "basic": r"(?i)(riemann surface|branch cut|covering|sheet)",
            "properties": r"(?i)(genus|compact|simply connected|uniformization)",
            "maps": r"(?i)(conformal map|automorphism|covering map|deck transformation)",
            "differential": r"(?i)(differential form|abelian|meromorphic form|period)"
        }
        self.special_functions = {
            "elementary": r"(?i)(exponential|logarithm|power|trigonometric)",
            "elliptic": r"(?i)(elliptic function|modular|weierstrass|jacobi)",
            "special": r"(?i)(gamma function|zeta function|theta function|hypergeometric)",
            "orthogonal": r"(?i)(legendre|bessel|hermite|chebyshev)"
        }
    
    def _initialize_structures(self) -> Dict[str, ComplexStructure]:
        """Initialize common complex analytic structures."""
        return {
            "local": ComplexStructure(
                name="Local Theory",
                properties={"holomorphic", "meromorphic", "conformal"},
                local_theory=["Power series", "Laurent series", "Residues"],
                global_theory=["Analytic continuation", "Monodromy"],
                applications=["Conformal mapping", "Physical applications"],
                theorems=["Cauchy's theorem", "Residue theorem", "Maximum modulus"]
            ),
            "global": ComplexStructure(
                name="Global Theory",
                properties={"compact", "connected", "simply connected"},
                local_theory=["Local coordinates", "Charts"],
                global_theory=["Riemann surfaces", "Covering spaces"],
                applications=["Algebraic curves", "Complex dynamics"],
                theorems=["Riemann mapping", "Uniformization", "Picard theorems"]
            ),
            "special": ComplexStructure(
                name="Special Functions",
                properties={"entire", "meromorphic", "periodic"},
                local_theory=["Series expansions", "Functional equations"],
                global_theory=["Analytic continuation", "Monodromy"],
                applications=["Mathematical physics", "Number theory"],
                theorems=["Identity theorem", "Reflection principle"]
            ),
            "geometric": ComplexStructure(
                name="Geometric Theory",
                properties={"conformal", "biholomorphic", "proper"},
                local_theory=["Conformal metrics", "Curvature"],
                global_theory=["Moduli spaces", "Teichmüller theory"],
                applications=["Complex geometry", "String theory"],
                theorems=["Schwarz lemma", "Montel's theorem"]
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the complex analysis model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.structures)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _check_holomorphic(self, text: str) -> Dict[str, Any]:
        """Analyze usage of holomorphic function concepts."""
        results = {
            "concepts_used": [],
            "missing_properties": [],
            "incorrect_usage": []
        }
        
        for concept_type, pattern in self.holomorphic.items():
            if re.search(pattern, text):
                results["concepts_used"].append(concept_type)
                
                # Check for proper properties
                if concept_type == "basic":
                    if not re.search(r"(?i)(cauchy-riemann|differentiable|analytic)", text):
                        results["missing_properties"].append("Basic holomorphicity properties not verified")
                
                elif concept_type == "singularities":
                    if not re.search(r"(?i)(laurent|residue|local behavior)", text):
                        results["missing_properties"].append("Singularity analysis not properly developed")
        
        return results
    
    def _analyze_integration(self, text: str) -> Dict[str, List[str]]:
        """Analyze complex integration concepts and properties."""
        integration = {
            "identified": [],
            "properties_verified": [],
            "issues": []
        }
        
        for int_type, pattern in self.integration.items():
            if re.search(pattern, text):
                integration["identified"].append(int_type)
                
                # Check for necessary properties
                if int_type == "contour":
                    if not re.search(r"(?i)(path|curve|oriented|closed)", text):
                        integration["issues"].append("Contour properties not properly specified")
                
                elif int_type == "residues":
                    if not re.search(r"(?i)(laurent|pole|order|computation)", text):
                        integration["issues"].append("Residue computation not properly established")
        
        return integration
    
    def _check_riemann_surfaces(self, text: str) -> Dict[str, Any]:
        """Analyze Riemann surface concepts."""
        results = {
            "surfaces": [],
            "properties": [],
            "missing_arguments": []
        }
        
        for surf_type, pattern in self.riemann_surfaces.items():
            if re.search(pattern, text):
                results["surfaces"].append(surf_type)
                
                # Check for specific requirements
                if surf_type == "basic":
                    if not re.search(r"(?i)(chart|atlas|transition|holomorphic)", text):
                        results["missing_arguments"].append("Riemann surface structure not properly defined")
                
                elif surf_type == "differential":
                    if not re.search(r"(?i)(holomorphic|meromorphic|residue|period)", text):
                        results["missing_arguments"].append("Differential form properties not established")
        
        return results
    
    def _analyze_special_functions(self, text: str) -> Dict[str, Any]:
        """Analyze special function theory."""
        results = {
            "functions": [],
            "properties": [],
            "relations": [],
            "issues": []
        }
        
        for func_type, pattern in self.special_functions.items():
            if re.search(pattern, text):
                results["functions"].append(func_type)
                
                # Check specific aspects
                if func_type == "elliptic":
                    if not re.search(r"(?i)(doubly periodic|lattice|order|pole)", text):
                        results["issues"].append("Elliptic function properties not properly verified")
                
                elif func_type == "special":
                    if not re.search(r"(?i)(functional equation|analytic continuation|pole)", text):
                        results["issues"].append("Special function properties not properly established")
        
        return results
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform complex analysis specific review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze different aspects
        holomorphic = self._check_holomorphic(content)
        integration = self._analyze_integration(content)
        surfaces = self._check_riemann_surfaces(content)
        functions = self._analyze_special_functions(content)
        
        # Generate comments and suggestions
        comments = []
        suggestions = []
        
        # Holomorphic analysis
        if holomorphic["concepts_used"]:
            comments.append(f"Holomorphic concepts: {', '.join(holomorphic['concepts_used'])}")
        if holomorphic["missing_properties"]:
            comments.extend(holomorphic["missing_properties"])
            suggestions.append("Verify holomorphicity properties")
        
        # Integration analysis
        if integration["identified"]:
            comments.append(f"Integration concepts: {', '.join(integration['identified'])}")
        if integration["issues"]:
            comments.extend(integration["issues"])
            suggestions.append("Complete integration arguments")
        
        # Riemann surface analysis
        if surfaces["surfaces"]:
            comments.append(f"Riemann surface aspects: {', '.join(surfaces['surfaces'])}")
        if surfaces["missing_arguments"]:
            comments.extend(surfaces["missing_arguments"])
            suggestions.append("Verify Riemann surface properties")
        
        # Special function analysis
        if functions["functions"]:
            comments.append(f"Special functions: {', '.join(functions['functions'])}")
        if functions["issues"]:
            comments.extend(functions["issues"])
            suggestions.append("Develop special function theory fully")
        
        # Calculate score
        score = 1.0
        score -= 0.1 * len(holomorphic["missing_properties"])
        score -= 0.1 * len(integration["issues"])
        score -= 0.1 * len(surfaces["missing_arguments"])
        score -= 0.1 * len(functions["issues"])
        score = max(0.0, score)
        
        # Calculate confidence
        confidence = 0.9 if (
            holomorphic["concepts_used"] or
            integration["identified"] or
            surfaces["surfaces"] or
            functions["functions"]
        ) else 0.7
        
        return ReviewResult(
            score=score,
            comments=comments,
            suggestions=suggestions,
            confidence=confidence,
            metadata={
                "holomorphic": holomorphic,
                "integration": integration,
                "riemann_surfaces": surfaces,
                "special_functions": functions
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing complex analytic insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "complex_analysis_score": review_result.score,
            "holomorphic_concepts": review_result.metadata["holomorphic"]["concepts_used"],
            "integration_methods": review_result.metadata["integration"]["identified"],
            "surface_aspects": review_result.metadata["riemann_surfaces"]["surfaces"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Complex Analysis paper review",
            "Holomorphic function verification",
            "Complex integration analysis",
            "Riemann surface assessment",
            "Special function verification",
            "Residue computations",
            "Mathematical rigor assessment",
            "Cross-field applications"
        ] 