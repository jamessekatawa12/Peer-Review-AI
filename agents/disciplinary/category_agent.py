from typing import Any, Dict, List, Set
import re
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class CategoryStructure:
    """Data class for storing categorical structures."""
    name: str
    objects: Set[str]
    morphisms: Set[str]
    properties: List[str]
    universal_properties: List[str]
    examples: List[str]
    theorems: List[str]

class CategoryTheoryAgent(BaseAgent):
    """Agent specialized in reviewing Category Theory papers."""
    
    def __init__(self, name: str = "Category Theory Reviewer", model_name: str = "allenai/mathbert-base-cased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.structures = self._initialize_structures()
        self.categorical_concepts = {
            "basic": r"(?i)(category|functor|natural transformation)",
            "limits": r"(?i)(limit|colimit|terminal|initial|product|coproduct)",
            "adjunctions": r"(?i)(adjoint|adjunction|unit|counit)",
            "universality": r"(?i)(universal property|universal arrow|universal object)"
        }
        self.diagram_patterns = {
            "commutative": r"(?i)(commute|diagram chase|commutative diagram)",
            "exact": r"(?i)(exact sequence|kernel|cokernel|image)",
            "pullback": r"(?i)(pullback|fiber product|cartesian)",
            "pushout": r"(?i)(pushout|amalgamated sum|cocartesian)"
        }
        self.advanced_concepts = {
            "monoidal": r"(?i)(monoidal|tensor|braided|symmetric)",
            "enriched": r"(?i)(enriched category|hom-object|V-category)",
            "topos": r"(?i)(topos|subobject classifier|cartesian closed)",
            "higher": r"(?i)(2-category|bicategory|∞-category|higher category)"
        }
        self.homological = {
            "abelian": r"(?i)(abelian category|additive|exact functor)",
            "derived": r"(?i)(derived category|derived functor|triangulated)",
            "spectral": r"(?i)(spectral sequence|filtration|graded object)",
            "cohomology": r"(?i)(cohomology|ext|tor|derived functor)"
        }
    
    def _initialize_structures(self) -> Dict[str, CategoryStructure]:
        """Initialize common categorical structures."""
        return {
            "category": CategoryStructure(
                name="Category",
                objects={"Objects", "Morphisms", "Identity"},
                morphisms={"Composition", "Identity morphism"},
                properties=["Associativity", "Identity laws"],
                universal_properties=["Initial/Terminal objects", "Products/Coproducts"],
                examples=["Set", "Top", "Grp", "Ring"],
                theorems=["Yoneda lemma", "Adjoint functor theorem"]
            ),
            "functor": CategoryStructure(
                name="Functor",
                objects={"Objects map", "Morphisms map"},
                morphisms={"Natural transformation", "Natural isomorphism"},
                properties=["Preserves composition", "Preserves identities"],
                universal_properties=["Left/Right adjoints", "Representable functors"],
                examples=["Forgetful functor", "Free functor", "Hom functor"],
                theorems=["Adjoint functor theorems", "Kan extension"]
            ),
            "abelian": CategoryStructure(
                name="Abelian Category",
                objects={"Kernels", "Cokernels", "Images"},
                morphisms={"Monomorphism", "Epimorphism", "Isomorphism"},
                properties=["Additive", "Has kernels/cokernels", "Normal/Conormal"],
                universal_properties=["Biproducts", "Pullbacks/Pushouts"],
                examples=["R-Mod", "Ab", "Sh(X)"],
                theorems=["Snake lemma", "Five lemma", "Long exact sequence"]
            ),
            "topos": CategoryStructure(
                name="Topos",
                objects={"Subobject classifier", "Power objects"},
                morphisms={"Geometric morphism", "Logical morphism"},
                properties=["Cartesian closed", "Has all limits", "Boolean"],
                universal_properties=["Exponentials", "Subobject classifier"],
                examples=["Set", "Sh(X)", "Presheaf categories"],
                theorems=["Giraud's theorem", "Diaconescu's theorem"]
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the category theory model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.structures)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _check_categorical_concepts(self, text: str) -> Dict[str, Any]:
        """Analyze usage of categorical concepts."""
        results = {
            "concepts_used": [],
            "missing_properties": [],
            "incorrect_usage": []
        }
        
        for concept_type, pattern in self.categorical_concepts.items():
            if re.search(pattern, text):
                results["concepts_used"].append(concept_type)
                
                # Check for proper properties
                if concept_type == "basic":
                    if not re.search(r"(?i)(associative|identity|composition)", text):
                        results["missing_properties"].append("Basic categorical properties not verified")
                
                elif concept_type == "adjunctions":
                    if not re.search(r"(?i)(unit|counit|natural|bijection)", text):
                        results["missing_properties"].append("Adjunction properties not properly established")
        
        return results
    
    def _analyze_diagrams(self, text: str) -> Dict[str, List[str]]:
        """Analyze categorical diagrams and their properties."""
        diagrams = {
            "identified": [],
            "properties_verified": [],
            "issues": []
        }
        
        for diag_type, pattern in self.diagram_patterns.items():
            if re.search(pattern, text):
                diagrams["identified"].append(diag_type)
                
                # Check for necessary verifications
                if diag_type == "commutative":
                    if not re.search(r"(?i)(check|verify|prove|commute)", text):
                        diagrams["issues"].append("Diagram commutativity not verified")
                
                elif diag_type == "exact":
                    if not re.search(r"(?i)(ker|im|sequence|exact)", text):
                        diagrams["issues"].append("Exactness not properly verified")
        
        return diagrams
    
    def _check_advanced_structures(self, text: str) -> Dict[str, Any]:
        """Analyze advanced categorical structures."""
        results = {
            "structures": [],
            "missing_axioms": [],
            "theorems_used": []
        }
        
        for struct_type, pattern in self.advanced_concepts.items():
            if re.search(pattern, text):
                results["structures"].append(struct_type)
                
                # Check for specific requirements
                if struct_type == "monoidal":
                    if not re.search(r"(?i)(associator|unitor|coherence)", text):
                        results["missing_axioms"].append("Monoidal category coherence conditions not verified")
                
                elif struct_type == "topos":
                    if not re.search(r"(?i)(subobject|cartesian|closed)", text):
                        results["missing_axioms"].append("Topos axioms not fully verified")
        
        return results
    
    def _analyze_homological(self, text: str) -> Dict[str, Any]:
        """Analyze homological algebra concepts."""
        results = {
            "concepts": [],
            "exact_sequences": [],
            "derived_functors": [],
            "issues": []
        }
        
        for concept_type, pattern in self.homological.items():
            if re.search(pattern, text):
                results["concepts"].append(concept_type)
                
                # Check specific aspects
                if concept_type == "abelian":
                    if not re.search(r"(?i)(kernel|cokernel|exact)", text):
                        results["issues"].append("Abelian category axioms not verified")
                
                elif concept_type == "derived":
                    if not re.search(r"(?i)(resolution|quasi|isomorphism)", text):
                        results["issues"].append("Derived functor construction not properly explained")
        
        return results
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform category theory specific review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze different aspects
        concepts = self._check_categorical_concepts(content)
        diagrams = self._analyze_diagrams(content)
        advanced = self._check_advanced_structures(content)
        homological = self._analyze_homological(content)
        
        # Generate comments and suggestions
        comments = []
        suggestions = []
        
        # Concept analysis
        if concepts["concepts_used"]:
            comments.append(f"Categorical concepts used: {', '.join(concepts['concepts_used'])}")
        if concepts["missing_properties"]:
            comments.extend(concepts["missing_properties"])
            suggestions.append("Verify all categorical properties and axioms")
        
        # Diagram analysis
        if diagrams["identified"]:
            comments.append(f"Categorical diagrams: {', '.join(diagrams['identified'])}")
        if diagrams["issues"]:
            comments.extend(diagrams["issues"])
            suggestions.append("Provide complete diagram chase arguments")
        
        # Advanced structure analysis
        if advanced["structures"]:
            comments.append(f"Advanced structures: {', '.join(advanced['structures'])}")
        if advanced["missing_axioms"]:
            comments.extend(advanced["missing_axioms"])
            suggestions.append("Verify all axioms for advanced categorical structures")
        
        # Homological analysis
        if homological["concepts"]:
            comments.append(f"Homological concepts: {', '.join(homological['concepts'])}")
        if homological["issues"]:
            comments.extend(homological["issues"])
            suggestions.append("Complete homological algebra arguments")
        
        # Calculate score
        score = 1.0
        score -= 0.1 * len(concepts["missing_properties"])
        score -= 0.1 * len(diagrams["issues"])
        score -= 0.1 * len(advanced["missing_axioms"])
        score -= 0.1 * len(homological["issues"])
        score = max(0.0, score)
        
        # Calculate confidence
        confidence = 0.9 if (
            concepts["concepts_used"] or
            diagrams["identified"] or
            advanced["structures"] or
            homological["concepts"]
        ) else 0.7
        
        return ReviewResult(
            score=score,
            comments=comments,
            suggestions=suggestions,
            confidence=confidence,
            metadata={
                "categorical_concepts": concepts,
                "diagrams": diagrams,
                "advanced_structures": advanced,
                "homological_analysis": homological
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing categorical insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "category_score": review_result.score,
            "concepts_used": review_result.metadata["categorical_concepts"]["concepts_used"],
            "advanced_structures": review_result.metadata["advanced_structures"]["structures"],
            "homological_concepts": review_result.metadata["homological_analysis"]["concepts"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Category Theory paper review",
            "Categorical concept verification",
            "Diagram chase checking",
            "Advanced structure analysis",
            "Homological algebra assessment",
            "Universal property verification",
            "Mathematical rigor assessment",
            "Cross-field categorical applications"
        ] 