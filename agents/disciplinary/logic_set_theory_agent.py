from typing import Any, Dict, List, Set, Optional
import re
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class LogicalStructure:
    """Data class for storing logical and set-theoretic structures."""
    name: str
    axioms: Set[str]
    rules: List[str]
    theorems: List[str]
    models: List[str]
    applications: List[str]

class LogicSetTheoryAgent(BaseAgent):
    """Agent specialized in reviewing Mathematical Logic and Set Theory papers."""
    
    def __init__(self, name: str = "Logic & Set Theory Reviewer", model_name: str = "allenai/mathbert-base-cased"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.structures = self._initialize_structures()
        self.formal_logic = {
            "propositional": r"(?i)(proposition|truth table|tautology|boolean)",
            "predicate": r"(?i)(predicate|quantifier|variable|formula)",
            "modal": r"(?i)(modal logic|necessity|possibility|accessibility)",
            "proof_theory": r"(?i)(proof|deduction|inference|sequent)"
        }
        self.set_theory = {
            "basic": r"(?i)(set|element|subset|union|intersection)",
            "cardinal": r"(?i)(cardinal|ordinal|transfinite|aleph)",
            "axioms": r"(?i)(zfc|choice|foundation|regularity|replacement)",
            "constructions": r"(?i)(powerset|product|quotient|relation)"
        }
        self.model_theory = {
            "basic": r"(?i)(model|structure|interpretation|satisfaction)",
            "constructions": r"(?i)(elementary|embedding|extension|submodel)",
            "properties": r"(?i)(complete|categorical|saturated|stable)",
            "methods": r"(?i)(compactness|lowenheim|ultraproduct|transfer)"
        }
        self.foundations = {
            "basic": r"(?i)(foundation|type theory|category theory|topos)",
            "constructive": r"(?i)(constructive|intuitionistic|computational|realizability)",
            "systems": r"(?i)(arithmetic|analysis|set theory|higher-order)",
            "metamath": r"(?i)(consistency|independence|completeness|decidability)"
        }
    
    def _initialize_structures(self) -> Dict[str, LogicalStructure]:
        """Initialize common logical and set-theoretic structures."""
        return {
            "first_order": LogicalStructure(
                name="First-Order Logic",
                axioms={"modus ponens", "generalization", "substitution"},
                rules=["Deduction", "Resolution", "Unification"],
                theorems=["Completeness", "Compactness", "Löwenheim-Skolem"],
                models=["Herbrand models", "Term models"],
                applications=["Formal verification", "Mathematics foundations"]
            ),
            "set_theoretic": LogicalStructure(
                name="Set Theory",
                axioms={"extensionality", "comprehension", "choice"},
                rules=["Class formation", "Transfinite induction"],
                theorems=["Cantor", "Zorn's lemma", "Well-ordering"],
                models=["Von Neumann universe", "Constructible universe"],
                applications=["Mathematics foundations", "Category theory"]
            ),
            "type_theoretic": LogicalStructure(
                name="Type Theory",
                axioms={"type formation", "term introduction", "computation"},
                rules=["Type checking", "Term reduction"],
                theorems=["Normalization", "Subject reduction"],
                models=["PER models", "Realizability models"],
                applications=["Programming languages", "Proof assistants"]
            ),
            "modal_logic": LogicalStructure(
                name="Modal Logic",
                axioms={"necessitation", "distribution", "reflexivity"},
                rules=["Modal generalization", "Box introduction"],
                theorems=["Completeness", "Canonicity"],
                models=["Kripke models", "Algebraic models"],
                applications=["Knowledge representation", "Temporal logic"]
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the logic and set theory model."""
        if not self.is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.structures)
            )
            self.model.eval()
            self.is_initialized = True
    
    def _check_formal_logic(self, text: str) -> Dict[str, Any]:
        """Analyze usage of formal logic concepts."""
        results = {
            "concepts_used": [],
            "missing_components": [],
            "incorrect_usage": []
        }
        
        for logic_type, pattern in self.formal_logic.items():
            if re.search(pattern, text):
                results["concepts_used"].append(logic_type)
                
                # Check for proper components
                if logic_type == "propositional":
                    if not re.search(r"(?i)(truth|validity|connective)", text):
                        results["missing_components"].append("Basic propositional components not specified")
                
                elif logic_type == "proof_theory":
                    if not re.search(r"(?i)(rule|axiom|theorem|lemma)", text):
                        results["missing_components"].append("Proof theoretical framework not established")
        
        return results
    
    def _analyze_set_theory(self, text: str) -> Dict[str, List[str]]:
        """Analyze set theory concepts and properties."""
        analysis = {
            "identified": [],
            "properties_verified": [],
            "issues": []
        }
        
        for set_type, pattern in self.set_theory.items():
            if re.search(pattern, text):
                analysis["identified"].append(set_type)
                
                # Check for necessary properties
                if set_type == "basic":
                    if not re.search(r"(?i)(membership|inclusion|operation)", text):
                        analysis["issues"].append("Basic set operations not properly defined")
                
                elif set_type == "cardinal":
                    if not re.search(r"(?i)(well-order|comparison|arithmetic)", text):
                        analysis["issues"].append("Cardinal arithmetic not properly established")
        
        return analysis
    
    def _check_model_theory(self, text: str) -> Dict[str, Any]:
        """Analyze model theory concepts."""
        results = {
            "models": [],
            "properties": [],
            "missing_arguments": []
        }
        
        for model_type, pattern in self.model_theory.items():
            if re.search(pattern, text):
                results["models"].append(model_type)
                
                # Check for specific requirements
                if model_type == "basic":
                    if not re.search(r"(?i)(language|signature|structure|domain)", text):
                        results["missing_arguments"].append("Model structure not properly defined")
                
                elif model_type == "properties":
                    if not re.search(r"(?i)(theory|model|satisfaction|truth)", text):
                        results["missing_arguments"].append("Model properties not properly established")
        
        return results
    
    def _analyze_foundations(self, text: str) -> Dict[str, Any]:
        """Analyze mathematical foundations."""
        results = {
            "systems": [],
            "properties": [],
            "relations": [],
            "issues": []
        }
        
        for found_type, pattern in self.foundations.items():
            if re.search(pattern, text):
                results["systems"].append(found_type)
                
                # Check specific aspects
                if found_type == "basic":
                    if not re.search(r"(?i)(axiom|primitive|definition|theorem)", text):
                        results["issues"].append("Foundational framework not properly specified")
                
                elif found_type == "systems":
                    if not re.search(r"(?i)(consistency|model|interpretation)", text):
                        results["issues"].append("System properties not properly established")
        
        return results
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform logic and set theory specific review of the manuscript."""
        if not self.is_initialized:
            await self.initialize()
        
        # Analyze different aspects
        logic = self._check_formal_logic(content)
        sets = self._analyze_set_theory(content)
        models = self._check_model_theory(content)
        foundations = self._analyze_foundations(content)
        
        # Generate comments and suggestions
        comments = []
        suggestions = []
        
        # Formal logic analysis
        if logic["concepts_used"]:
            comments.append(f"Logical concepts: {', '.join(logic['concepts_used'])}")
        if logic["missing_components"]:
            comments.extend(logic["missing_components"])
            suggestions.append("Complete logical framework")
        
        # Set theory analysis
        if sets["identified"]:
            comments.append(f"Set theory concepts: {', '.join(sets['identified'])}")
        if sets["issues"]:
            comments.extend(sets["issues"])
            suggestions.append("Verify set-theoretic foundations")
        
        # Model theory analysis
        if models["models"]:
            comments.append(f"Model theory aspects: {', '.join(models['models'])}")
        if models["missing_arguments"]:
            comments.extend(models["missing_arguments"])
            suggestions.append("Complete model-theoretic arguments")
        
        # Foundations analysis
        if foundations["systems"]:
            comments.append(f"Foundational systems: {', '.join(foundations['systems'])}")
        if foundations["issues"]:
            comments.extend(foundations["issues"])
            suggestions.append("Strengthen foundational framework")
        
        # Calculate score
        score = 1.0
        score -= 0.1 * len(logic["missing_components"])
        score -= 0.1 * len(sets["issues"])
        score -= 0.1 * len(models["missing_arguments"])
        score -= 0.1 * len(foundations["issues"])
        score = max(0.0, score)
        
        # Calculate confidence
        confidence = 0.9 if (
            logic["concepts_used"] or
            sets["identified"] or
            models["models"] or
            foundations["systems"]
        ) else 0.7
        
        return ReviewResult(
            score=score,
            comments=comments,
            suggestions=suggestions,
            confidence=confidence,
            metadata={
                "formal_logic": logic,
                "set_theory": sets,
                "model_theory": models,
                "foundations": foundations
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing logical and set-theoretic insights."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "logic_score": review_result.score,
            "logical_concepts": review_result.metadata["formal_logic"]["concepts_used"],
            "set_theory_aspects": review_result.metadata["set_theory"]["identified"],
            "model_theory_aspects": review_result.metadata["model_theory"]["models"],
            "recommendations": review_result.suggestions
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Logic & Set Theory paper review",
            "Formal logic verification",
            "Set theory analysis",
            "Model theory assessment",
            "Mathematical foundations",
            "Proof verification",
            "Logical rigor assessment",
            "Cross-field applications"
        ] 