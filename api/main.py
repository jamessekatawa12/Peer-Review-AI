from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio

from ..agents.base.agent import ReviewResult
from ..agents.ethics.ethics_agent import EthicsAgent
from ..agents.disciplinary.methodology_agent import MethodologyAgent
from ..agents.disciplinary.citation_agent import CitationAgent
from ..agents.disciplinary.differential_geometry_agent import DifferentialGeometryAgent
from ..agents.disciplinary.category_agent import CategoryTheoryAgent
from ..agents.disciplinary.algebraic_geometry_agent import AlgebraicGeometryAgent
from ..agents.disciplinary.number_theory_agent import NumberTheoryAgent
from ..agents.disciplinary.functional_analysis_agent import FunctionalAnalysisAgent
from ..agents.disciplinary.probability_agent import ProbabilityTheoryAgent
from ..agents.disciplinary.complex_analysis_agent import ComplexAnalysisAgent
from ..core.knowledge_graph import KnowledgeGraph

app = FastAPI(title="AI Peer Review System")

# Initialize agents
ethics_agent = EthicsAgent()
methodology_agent = MethodologyAgent()
citation_agent = CitationAgent()
knowledge_graph = KnowledgeGraph()

# Initialize mathematical agents
differential_geometry_agent = DifferentialGeometryAgent()
category_theory_agent = CategoryTheoryAgent()
algebraic_geometry_agent = AlgebraicGeometryAgent()
number_theory_agent = NumberTheoryAgent()
functional_analysis_agent = FunctionalAnalysisAgent()
probability_theory_agent = ProbabilityTheoryAgent()
complex_analysis_agent = ComplexAnalysisAgent()

class PaperSubmission(BaseModel):
    title: str
    abstract: str
    content: str
    field: str
    references: List[str]
    metadata: Dict[str, Any]

class ReviewResponse(BaseModel):
    overall_score: float
    ethics_review: ReviewResult
    methodology_review: ReviewResult
    citation_review: ReviewResult
    field_specific_review: Optional[ReviewResult]
    knowledge_graph_analysis: Dict[str, Any]
    recommendations: List[str]

@app.post("/review", response_model=ReviewResponse)
async def review_paper(submission: PaperSubmission):
    try:
        # Initialize all agents if needed
        await asyncio.gather(
            ethics_agent.initialize(),
            methodology_agent.initialize(),
            citation_agent.initialize(),
            differential_geometry_agent.initialize(),
            category_theory_agent.initialize(),
            algebraic_geometry_agent.initialize(),
            number_theory_agent.initialize(),
            functional_analysis_agent.initialize(),
            probability_theory_agent.initialize(),
            complex_analysis_agent.initialize()
        )

        # Perform basic reviews
        ethics_review = await ethics_agent.review(submission.content, submission.metadata)
        methodology_review = await methodology_agent.review(submission.content, submission.metadata)
        citation_review = await citation_agent.review(submission.content, {"references": submission.references})

        # Field-specific review based on paper field
        field_specific_review = None
        if submission.field.lower() in ["differential geometry", "geometry"]:
            field_specific_review = await differential_geometry_agent.review(submission.content, submission.metadata)
        elif submission.field.lower() in ["category theory", "categories"]:
            field_specific_review = await category_theory_agent.review(submission.content, submission.metadata)
        elif submission.field.lower() in ["algebraic geometry", "schemes"]:
            field_specific_review = await algebraic_geometry_agent.review(submission.content, submission.metadata)
        elif submission.field.lower() in ["number theory", "arithmetic"]:
            field_specific_review = await number_theory_agent.review(submission.content, submission.metadata)
        elif submission.field.lower() in ["functional analysis", "analysis"]:
            field_specific_review = await functional_analysis_agent.review(submission.content, submission.metadata)
        elif submission.field.lower() in ["probability", "stochastic"]:
            field_specific_review = await probability_theory_agent.review(submission.content, submission.metadata)
        elif submission.field.lower() in ["complex analysis", "complex variables"]:
            field_specific_review = await complex_analysis_agent.review(submission.content, submission.metadata)

        # Update knowledge graph
        graph_analysis = knowledge_graph.analyze_paper(
            submission.content,
            submission.references,
            submission.field
        )

        # Calculate overall score
        base_score = (
            ethics_review.score * 0.3 +
            methodology_review.score * 0.3 +
            citation_review.score * 0.2
        )
        
        if field_specific_review:
            overall_score = base_score + (field_specific_review.score * 0.2)
        else:
            overall_score = base_score / 0.8  # Normalize to 1.0 scale

        # Compile recommendations
        recommendations = []
        recommendations.extend(ethics_review.suggestions)
        recommendations.extend(methodology_review.suggestions)
        recommendations.extend(citation_review.suggestions)
        if field_specific_review:
            recommendations.extend(field_specific_review.suggestions)

        return ReviewResponse(
            overall_score=overall_score,
            ethics_review=ethics_review,
            methodology_review=methodology_review,
            citation_review=citation_review,
            field_specific_review=field_specific_review,
            knowledge_graph_analysis=graph_analysis,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/capabilities")
async def get_agent_capabilities():
    """Get the capabilities of all available agents."""
    return {
        "ethics": ethics_agent.get_capabilities(),
        "methodology": methodology_agent.get_capabilities(),
        "citation": citation_agent.get_capabilities(),
        "differential_geometry": differential_geometry_agent.get_capabilities(),
        "category_theory": category_theory_agent.get_capabilities(),
        "algebraic_geometry": algebraic_geometry_agent.get_capabilities(),
        "number_theory": number_theory_agent.get_capabilities(),
        "functional_analysis": functional_analysis_agent.get_capabilities(),
        "probability_theory": probability_theory_agent.get_capabilities(),
        "complex_analysis": complex_analysis_agent.get_capabilities()
    }

@app.post("/collaborate/{agent_type}")
async def agent_collaboration(agent_type: str, data: Dict[str, Any]):
    """Enable collaboration between different types of agents."""
    try:
        if agent_type == "mathematics":
            # Collaborate between mathematical agents
            results = await asyncio.gather(
                differential_geometry_agent.collaborate(category_theory_agent, data),
                algebraic_geometry_agent.collaborate(number_theory_agent, data),
                functional_analysis_agent.collaborate(probability_theory_agent, data),
                complex_analysis_agent.collaborate(functional_analysis_agent, data)
            )
            return {"collaboration_results": results}
        elif agent_type == "methodology":
            # Collaborate between methodology and field-specific agents
            results = await asyncio.gather(
                methodology_agent.collaborate(differential_geometry_agent, data),
                methodology_agent.collaborate(complex_analysis_agent, data)
            )
            return {"collaboration_results": results}
        else:
            raise HTTPException(status_code=400, detail="Invalid agent type for collaboration")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check the health status of the API and its components."""
    return {
        "status": "healthy",
        "agents": {
            "ethics": ethics_agent.is_initialized,
            "methodology": methodology_agent.is_initialized,
            "citation": citation_agent.is_initialized,
            "differential_geometry": differential_geometry_agent.is_initialized,
            "category_theory": category_theory_agent.is_initialized,
            "algebraic_geometry": algebraic_geometry_agent.is_initialized,
            "number_theory": number_theory_agent.is_initialized,
            "functional_analysis": functional_analysis_agent.is_initialized,
            "probability_theory": probability_theory_agent.is_initialized,
            "complex_analysis": complex_analysis_agent.is_initialized
        },
        "knowledge_graph": "active"
    }
