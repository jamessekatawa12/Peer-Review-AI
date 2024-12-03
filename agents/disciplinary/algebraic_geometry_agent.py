from typing import Any, Dict, List, Set
import re
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class SchemeStructure:
    """Data class for storing scheme-theoretic structures."""
    name: str
    properties: Set[str]
    local_structure: List[str]
    morphisms: List[str]
    cohomology: List[str]
    theorems: List[str]

class AlgebraicGeometryAgent(BaseAgent):
    """Agent specialized in reviewing Algebraic Geometry papers."""
    
    def __init__(self, name: str = "Algebraic Geometry Reviewer", model_name: str = "allenai/mathbert-base-cased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.structures = self._initialize_structures()
        self.scheme_concepts = {
            "basic": r"(?i)(scheme|variety|spec|locally ringed space)",
            "properties": r"(?i)(reduced|irreducible|normal|regular|smooth)",
            "morphisms": r"(?i)(morphism of schemes|proper|flat|étale|finite)",
            "operations": r"(?i)(fiber product|base change|descent|blow-up)"
        }
        self.sheaf_concepts = {
            "coherent": r"(?i)(coherent sheaf|quasi-coherent|vector bundle)",
            "operations": r"(?i)(pushforward|pullback|tensor product|hom)",
            "resolutions": r"(?i)(resolution|locally free|injective|projective)",
            "derived": r"(?i)(derived functor|ext|tor|local cohomology)"
        }
        self.cohomology_theories = {
            "sheaf": r"(?i)(sheaf cohomology|H\^[0-9]|čech|derived functor)",
            "de_rham": r"(?i)(de rham|differential forms|hodge|crystalline)",
            "etale": r"(?i)(étale cohomology|l-adic|galois|fundamental group)",
            "intersection": r"(?i)(intersection theory|chow|cycle|K-theory)"
        }
        self.arithmetic = {
            "numbers": r"(?i)(number field|arithmetic|rational points|height)",
            "moduli": r"(?i)(moduli space|stack|deformation|family)",
            "reduction": r"(?i)(reduction mod p|good reduction|bad reduction)",
            "zeta": r"(?i)(zeta function|L-function|euler product|functional equation)"
        }
    
    def _initialize_structures(self) -> Dict[str, SchemeStructure]:
        """Initialize common scheme-theoretic structures."""
        return {
            "affine": SchemeStructure(
                name="Affine Scheme",
                properties={"locally ringed space", "spectrum", "prime ideal"},
                local_structure=["Ring of functions", "Local rings", "Points"],
                morphisms=["Ring homomorphisms", "Spec map"],
                cohomology=["Čech cohomology", "Local cohomology"],
                theorems=["Hilbert Nullstellensatz", "Serre's criterion"]
            ),
            "projective": SchemeStructure(
                name="Projective Scheme",
                properties={"proper", "graded", "ample"},
                local_structure=["Homogeneous coordinates", "Line bundles"],
                morphisms=["Projective morphisms", "Veronese embedding"],
                cohomology=["Serre cohomology", "Hodge theory"],
                theorems=["Serre duality", "Kodaira vanishing"]
            ),
            "curves": SchemeStructure(
                name="Algebraic Curve",
                properties={"smooth", "genus", "degree"},
                local_structure=["Local parameters", "Divisors"],
                morphisms=["Ramification", "Degree"],
                cohomology=["Riemann-Roch", "Serre duality"],
                theorems=["Riemann-Roch theorem", "Hurwitz formula"]
            ),
            "surfaces": SchemeStructure(
                name="Algebraic Surface",
                properties={"Kodaira dimension", "rational", "ruled"},
                local_structure=["Intersection form", "Exceptional curves"],
                morphisms=["Birational maps", "Minimal models"],
                cohomology=["Hodge diamond", "Noether formula"],
                theorems=["Enriques classification", "Castelnuovo criterion"]
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the algebraic geometry model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.structures)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _check_scheme_concepts(self, text: str) -> Dict[str, Any]:
        """Analyze usage of scheme-theoretic concepts."""
        results = {
            "concepts_used": [],
            "missing_properties": [],
            "incorrect_usage": []
        }
        
        for concept_type, pattern in self.scheme_concepts.items():
            if re.search(pattern, text):
                results["concepts_used"].append(concept_type)
                
                # Check for proper properties
                if concept_type == "basic":
                    if not re.search(r"(?i)(locally ringed|structure sheaf|spec)", text):
                        results["missing_properties"].append("Basic scheme properties not verified")
                
                elif concept_type == "morphisms":
                    if not re.search(r"(?i)(commutative|functorial|base change)", text):
                        results["missing_properties"].append("Morphism properties not properly established")
        
        return results
    
    def _analyze_sheaves(self, text: str) -> Dict[str, List[str]]:
        """Analyze sheaf theory concepts and properties."""
        sheaves = {
            "identified": [],
            "properties_verified": [],
            "issues": []
        }
        
        for sheaf_type, pattern in self.sheaf_concepts.items():
            if re.search(pattern, text):
                sheaves["identified"].append(sheaf_type)
                
                # Check for necessary properties
                if sheaf_type == "coherent":
                    if not re.search(r"(?i)(finitely generated|locally|noetherian)", text):
                        sheaves["issues"].append("Coherence conditions not verified")
                
                elif sheaf_type == "resolutions":
                    if not re.search(r"(?i)(exact sequence|acyclic|resolution)", text):
                        sheaves["issues"].append("Resolution properties not properly verified")
        
        return sheaves
    
    def _check_cohomology(self, text: str) -> Dict[str, Any]:
        """Analyze cohomology computations and properties."""
        results = {
            "theories_used": [],
            "computations": [],
            "missing_arguments": []
        }
        
        for theory_type, pattern in self.cohomology_theories.items():
            if re.search(pattern, text):
                results["theories_used"].append(theory_type)
                
                # Check for specific requirements
                if theory_type == "sheaf":
                    if not re.search(r"(?i)(spectral sequence|derived|acyclic)", text):
                        results["missing_arguments"].append("Cohomology computation details missing")
                
                elif theory_type == "intersection":
                    if not re.search(r"(?i)(proper|intersection|cycle class)", text):
                        results["missing_arguments"].append("Intersection theory foundations not established")
        
        return results
    
    def _analyze_arithmetic(self, text: str) -> Dict[str, Any]:
        """Analyze arithmetic aspects of schemes."""
        results = {
            "aspects": [],
            "local_analysis": [],
            "global_properties": [],
            "issues": []
        }
        
        for aspect_type, pattern in self.arithmetic.items():
            if re.search(pattern, text):
                results["aspects"].append(aspect_type)
                
                # Check specific aspects
                if aspect_type == "numbers":
                    if not re.search(r"(?i)(field extension|galois|splitting)", text):
                        results["issues"].append("Number field properties not verified")
                
                elif aspect_type == "moduli":
                    if not re.search(r"(?i)(universal|representable|functor)", text):
                        results["issues"].append("Moduli problem not properly formulated")
        
        return results
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform algebraic geometry specific review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze different aspects
        schemes = self._check_scheme_concepts(content)
        sheaves = self._analyze_sheaves(content)
        cohomology = self._check_cohomology(content)
        arithmetic = self._analyze_arithmetic(content)
        
        # Generate comments and suggestions
        comments = []
        suggestions = []
        
        # Scheme analysis
        if schemes["concepts_used"]:
            comments.append(f"Scheme concepts used: {', '.join(schemes['concepts_used'])}")
        if schemes["missing_properties"]:
            comments.extend(schemes["missing_properties"])
            suggestions.append("Verify all scheme-theoretic properties")
        
        # Sheaf analysis
        if sheaves["identified"]:
            comments.append(f"Sheaf concepts: {', '.join(sheaves['identified'])}")
        if sheaves["issues"]:
            comments.extend(sheaves["issues"])
            suggestions.append("Complete sheaf-theoretic arguments")
        
        # Cohomology analysis
        if cohomology["theories_used"]:
            comments.append(f"Cohomology theories: {', '.join(cohomology['theories_used'])}")
        if cohomology["missing_arguments"]:
            comments.extend(cohomology["missing_arguments"])
            suggestions.append("Provide complete cohomology computations")
        
        # Arithmetic analysis
        if arithmetic["aspects"]:
            comments.append(f"Arithmetic aspects: {', '.join(arithmetic['aspects'])}")
        if arithmetic["issues"]:
            comments.extend(arithmetic["issues"])
            suggestions.append("Complete arithmetic arguments")
        
        # Calculate score
        score = 1.0
        score -= 0.1 * len(schemes["missing_properties"])
        score -= 0.1 * len(sheaves["issues"])
        score -= 0.1 * len(cohomology["missing_arguments"])
        score -= 0.1 * len(arithmetic["issues"])
        score = max(0.0, score)
        
        # Calculate confidence
        confidence = 0.9 if (
            schemes["concepts_used"] or
            sheaves["identified"] or
            cohomology["theories_used"] or
            arithmetic["aspects"]
        ) else 0.7
        
        return ReviewResult(
            score=score,
            comments=comments,
            suggestions=suggestions,
            confidence=confidence,
            metadata={
                "scheme_concepts": schemes,
                "sheaf_theory": sheaves,
                "cohomology": cohomology,
                "arithmetic": arithmetic
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing algebraic geometric insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "algebraic_geometry_score": review_result.score,
            "scheme_concepts": review_result.metadata["scheme_concepts"]["concepts_used"],
            "sheaf_concepts": review_result.metadata["sheaf_theory"]["identified"],
            "cohomology_theories": review_result.metadata["cohomology"]["theories_used"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Algebraic Geometry paper review",
            "Scheme theory verification",
            "Sheaf theory analysis",
            "Cohomology computations",
            "Arithmetic geometry assessment",
            "Intersection theory verification",
            "Mathematical rigor assessment",
            "Cross-field applications"
        ] 