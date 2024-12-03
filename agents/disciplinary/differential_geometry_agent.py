from typing import Any, Dict, List, Set
import re
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class GeometricStructure:
    """Data class for storing differential geometric structures."""
    name: str
    properties: Set[str]
    local_coordinates: List[str]
    global_properties: List[str]
    invariants: List[str]
    theorems: List[str]

class DifferentialGeometryAgent(BaseAgent):
    """Agent specialized in reviewing Differential Geometry papers."""
    
    def __init__(self, name: str = "Differential Geometry Reviewer", model_name: str = "allenai/mathbert-base-cased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.structures = self._initialize_structures()
        self.differential_operators = {
            "exterior_derivative": r"(?i)(d[^a-zA-Z]|exterior derivative|d\omega)",
            "covariant_derivative": r"(?i)(∇|covariant derivative|parallel transport)",
            "lie_derivative": r"(?i)(£|lie derivative|L_X)",
            "laplacian": r"(?i)(Δ|laplacian|laplace-beltrami)"
        }
        self.geometric_objects = {
            "tensors": r"(?i)(tensor|[a-z]_[{]?[ij][}]?|[a-z]\^[{]?[ij][}]?)",
            "forms": r"(?i)(differential form|[n]-form|\wedge|\omega\^n)",
            "vector_fields": r"(?i)(vector field|X\(M\)|tangent|∂_[{]?[i][}]?)",
            "metrics": r"(?i)(metric|g_[{]?[ij][}]?|riemannian|pseudo-riemannian)"
        }
        self.curvature_concepts = {
            "riemann": r"(?i)(riemann|R[{]?[ijkl][}]?|curvature tensor)",
            "ricci": r"(?i)(ricci|Ric|scalar curvature|R[{]?[ij][}]?)",
            "sectional": r"(?i)(sectional curvature|K\(σ\)|gaussian)",
            "scalar": r"(?i)(scalar curvature|R(?![{])|mean curvature)"
        }
        self.fiber_bundles = {
            "bundle_maps": r"(?i)(bundle map|section|fiber|π:E→B)",
            "connections": r"(?i)(connection|horizontal|vertical|ehresmann)",
            "characteristic_classes": r"(?i)(chern class|pontryagin|euler class)",
            "holonomy": r"(?i)(holonomy|monodromy|parallel transport)"
        }
    
    def _initialize_structures(self) -> Dict[str, GeometricStructure]:
        """Initialize common differential geometric structures."""
        return {
            "manifold": GeometricStructure(
                name="Differentiable Manifold",
                properties={"smooth", "orientable", "paracompact"},
                local_coordinates=["Charts", "Transition functions", "Atlas"],
                global_properties=["Topology", "Differential structure"],
                invariants=["de Rham cohomology", "Characteristic classes"],
                theorems=["Whitney embedding", "Frobenius theorem"]
            ),
            "riemannian": GeometricStructure(
                name="Riemannian Manifold",
                properties={"metric", "isometric", "complete"},
                local_coordinates=["Normal coordinates", "Exponential map"],
                global_properties=["Geodesic completeness", "Curvature"],
                invariants=["Volume form", "Hodge star", "Laplacian"],
                theorems=["Hopf-Rinow", "Bonnet-Myers", "Gauss-Bonnet"]
            ),
            "bundle": GeometricStructure(
                name="Fiber Bundle",
                properties={"local triviality", "compatibility"},
                local_coordinates=["Local trivializations", "Transition functions"],
                global_properties=["Total space", "Base space", "Fiber"],
                invariants=["Characteristic classes", "Connection forms"],
                theorems=["Classification of bundles", "Ehresmann's theorem"]
            ),
            "symplectic": GeometricStructure(
                name="Symplectic Manifold",
                properties={"closed", "non-degenerate"},
                local_coordinates=["Darboux coordinates"],
                global_properties=["Symplectic form", "Poisson structure"],
                invariants=["Symplectic volume", "Gromov width"],
                theorems=["Darboux theorem", "Moser's theorem"]
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the differential geometry model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.structures)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _check_differential_operators(self, text: str) -> Dict[str, Any]:
        """Analyze usage of differential operators."""
        results = {
            "operators_used": [],
            "missing_properties": [],
            "incorrect_usage": []
        }
        
        for op_type, pattern in self.differential_operators.items():
            if re.search(pattern, text):
                results["operators_used"].append(op_type)
                
                # Check for proper properties
                if op_type == "exterior_derivative":
                    if not re.search(r"(?i)(d²\s*=\s*0|closed|exact)", text):
                        results["missing_properties"].append("d² = 0 property or closed/exact forms not discussed")
                
                elif op_type == "covariant_derivative":
                    if not re.search(r"(?i)(leibniz|metric compatible|torsion)", text):
                        results["missing_properties"].append("Covariant derivative properties not verified")
        
        return results
    
    def _analyze_geometric_objects(self, text: str) -> Dict[str, List[str]]:
        """Analyze geometric objects and their properties."""
        objects = {
            "identified": [],
            "properties_verified": [],
            "issues": []
        }
        
        for obj_type, pattern in self.geometric_objects.items():
            if re.search(pattern, text):
                objects["identified"].append(obj_type)
                
                # Check for necessary properties
                if obj_type == "metrics":
                    if not re.search(r"(?i)(positive definite|non-degenerate|signature)", text):
                        objects["issues"].append("Metric properties not specified")
                
                elif obj_type == "forms":
                    if not re.search(r"(?i)(closed|exact|degree|wedge product)", text):
                        objects["issues"].append("Differential form properties not discussed")
        
        return objects
    
    def _check_curvature(self, text: str) -> Dict[str, Any]:
        """Analyze curvature-related concepts and calculations."""
        results = {
            "curvature_types": [],
            "properties_verified": [],
            "missing_arguments": []
        }
        
        for curv_type, pattern in self.curvature_concepts.items():
            if re.search(pattern, text):
                results["curvature_types"].append(curv_type)
                
                # Check for specific requirements
                if curv_type == "riemann":
                    if not re.search(r"(?i)(symmetr|bianchi|tensor)", text):
                        results["missing_arguments"].append("Riemann tensor symmetries not verified")
                
                elif curv_type == "ricci":
                    if not re.search(r"(?i)(trace|contract|symmetric)", text):
                        results["missing_arguments"].append("Ricci curvature properties not discussed")
        
        return results
    
    def _analyze_bundles(self, text: str) -> Dict[str, Any]:
        """Analyze fiber bundle structures and properties."""
        results = {
            "bundle_types": [],
            "local_properties": [],
            "global_properties": [],
            "issues": []
        }
        
        for bundle_aspect, pattern in self.fiber_bundles.items():
            if re.search(pattern, text):
                results["bundle_types"].append(bundle_aspect)
                
                # Check local vs global properties
                if bundle_aspect == "bundle_maps":
                    if not re.search(r"(?i)(local trivialization|transition|compatible)", text):
                        results["issues"].append("Bundle local triviality not verified")
                
                elif bundle_aspect == "connections":
                    if not re.search(r"(?i)(horizontal|vertical|splitting)", text):
                        results["issues"].append("Connection properties not properly defined")
        
        return results
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform differential geometry specific review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze different aspects
        operators = self._check_differential_operators(content)
        objects = self._analyze_geometric_objects(content)
        curvature = self._check_curvature(content)
        bundles = self._analyze_bundles(content)
        
        # Generate comments and suggestions
        comments = []
        suggestions = []
        
        # Operator analysis
        if operators["operators_used"]:
            comments.append(f"Differential operators used: {', '.join(operators['operators_used'])}")
        if operators["missing_properties"]:
            comments.extend(operators["missing_properties"])
            suggestions.append("Verify all properties of differential operators")
        
        # Geometric objects analysis
        if objects["identified"]:
            comments.append(f"Geometric objects analyzed: {', '.join(objects['identified'])}")
        if objects["issues"]:
            comments.extend(objects["issues"])
            suggestions.append("Specify properties of geometric objects completely")
        
        # Curvature analysis
        if curvature["curvature_types"]:
            comments.append(f"Curvature concepts: {', '.join(curvature['curvature_types'])}")
        if curvature["missing_arguments"]:
            comments.extend(curvature["missing_arguments"])
            suggestions.append("Complete curvature-related arguments")
        
        # Bundle analysis
        if bundles["bundle_types"]:
            comments.append(f"Bundle structures: {', '.join(bundles['bundle_types'])}")
        if bundles["issues"]:
            comments.extend(bundles["issues"])
            suggestions.append("Verify bundle properties both locally and globally")
        
        # Calculate score
        score = 1.0
        score -= 0.1 * len(operators["missing_properties"])
        score -= 0.1 * len(objects["issues"])
        score -= 0.1 * len(curvature["missing_arguments"])
        score -= 0.1 * len(bundles["issues"])
        score = max(0.0, score)
        
        # Calculate confidence
        confidence = 0.9 if (
            operators["operators_used"] or
            objects["identified"] or
            curvature["curvature_types"] or
            bundles["bundle_types"]
        ) else 0.7
        
        return ReviewResult(
            score=score,
            comments=comments,
            suggestions=suggestions,
            confidence=confidence,
            metadata={
                "differential_operators": operators,
                "geometric_objects": objects,
                "curvature_analysis": curvature,
                "bundle_analysis": bundles
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing geometric insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "geometry_score": review_result.score,
            "operators_used": review_result.metadata["differential_operators"]["operators_used"],
            "geometric_objects": review_result.metadata["geometric_objects"]["identified"],
            "curvature_types": review_result.metadata["curvature_analysis"]["curvature_types"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Differential Geometry paper review",
            "Differential operator analysis",
            "Geometric object verification",
            "Curvature computation checking",
            "Fiber bundle analysis",
            "Local-to-global principle verification",
            "Mathematical rigor assessment",
            "Cross-field geometric applications"
        ] 