from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio

from ..agents.disciplinary.plagiarism_agent import PlagiarismDetectionAgent
from ..agents.ethics.ethics_agent import EthicsReviewAgent
from ..agents.disciplinary.methodology_agent import MethodologyAnalysisAgent
from ..agents.disciplinary.citation_agent import CitationVerificationAgent
from ..core.knowledge_graph import KnowledgeGraph

app = FastAPI(
    title="Peer Review AI API",
    description="API for the AI Multi-Agent Peer Review Framework",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents and knowledge graph
plagiarism_agent = PlagiarismDetectionAgent()
ethics_agent = EthicsReviewAgent()
methodology_agent = MethodologyAnalysisAgent()
citation_agent = CitationVerificationAgent()
knowledge_graph = KnowledgeGraph()

class ManuscriptSubmission(BaseModel):
    title: str
    abstract: str
    content: str
    authors: List[str]
    keywords: List[str]
    discipline: str

class ReviewResponse(BaseModel):
    manuscript_id: str
    overall_score: float
    agent_reviews: Dict[str, Any]
    interdisciplinary_insights: Dict[str, Any]
    suggestions: List[str]
    status: str

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup."""
    await asyncio.gather(
        plagiarism_agent.initialize(),
        ethics_agent.initialize(),
        methodology_agent.initialize(),
        citation_agent.initialize()
    )

@app.post("/api/v1/submit", response_model=ReviewResponse)
async def submit_manuscript(manuscript: ManuscriptSubmission):
    """Submit a manuscript for review."""
    try:
        # Perform parallel agent reviews
        review_tasks = [
            plagiarism_agent.review(manuscript.content, {"title": manuscript.title}),
            ethics_agent.review(manuscript.content, {"discipline": manuscript.discipline}),
            methodology_agent.review(manuscript.content, {"discipline": manuscript.discipline}),
            citation_agent.review(manuscript.content, {})
        ]
        
        review_results = await asyncio.gather(*review_tasks)
        plagiarism_result, ethics_result, methodology_result, citation_result = review_results
        
        # Analyze interdisciplinary connections
        knowledge_graph_analysis = knowledge_graph.analyze_manuscript(manuscript.content)
        
        # Calculate overall score (weighted average)
        weights = {
            "plagiarism": 0.25,
            "ethics": 0.25,
            "methodology": 0.25,
            "citations": 0.25
        }
        
        overall_score = sum([
            plagiarism_result.score * weights["plagiarism"],
            ethics_result.score * weights["ethics"],
            methodology_result.score * weights["methodology"],
            citation_result.score * weights["citations"]
        ])
        
        # Combine all suggestions
        all_suggestions = []
        all_suggestions.extend(plagiarism_result.suggestions)
        all_suggestions.extend(ethics_result.suggestions)
        all_suggestions.extend(methodology_result.suggestions)
        all_suggestions.extend(citation_result.suggestions)
        
        return ReviewResponse(
            manuscript_id="MS" + str(hash(manuscript.title))[:8],
            overall_score=overall_score,
            agent_reviews={
                "plagiarism": {
                    "score": plagiarism_result.score,
                    "comments": plagiarism_result.comments,
                    "confidence": plagiarism_result.confidence
                },
                "ethics": {
                    "score": ethics_result.score,
                    "comments": ethics_result.comments,
                    "confidence": ethics_result.confidence
                },
                "methodology": {
                    "score": methodology_result.score,
                    "comments": methodology_result.comments,
                    "confidence": methodology_result.confidence
                },
                "citations": {
                    "score": citation_result.score,
                    "comments": citation_result.comments,
                    "confidence": citation_result.confidence
                }
            },
            interdisciplinary_insights=knowledge_graph_analysis,
            suggestions=all_suggestions,
            status="completed"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF manuscript."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # In a real implementation, we would:
    # 1. Save the PDF
    # 2. Extract text using a PDF parser
    # 3. Process the extracted text
    
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/v1/status/{manuscript_id}")
async def get_review_status(manuscript_id: str):
    """Get the status of a manuscript review."""
    # In a real implementation, we would check a database
    return {"manuscript_id": manuscript_id, "status": "in_progress"}

@app.get("/api/v1/agents")
async def list_agents():
    """List available review agents and their capabilities."""
    return {
        "agents": [
            {
                "name": plagiarism_agent.name,
                "type": plagiarism_agent.agent_type.value,
                "capabilities": plagiarism_agent.get_capabilities()
            },
            {
                "name": ethics_agent.name,
                "type": ethics_agent.agent_type.value,
                "capabilities": ethics_agent.get_capabilities()
            },
            {
                "name": methodology_agent.name,
                "type": methodology_agent.agent_type.value,
                "capabilities": methodology_agent.get_capabilities()
            },
            {
                "name": citation_agent.name,
                "type": citation_agent.agent_type.value,
                "capabilities": citation_agent.get_capabilities()
            }
        ]
    }

@app.get("/api/v1/knowledge-graph/concepts/{concept}")
async def get_concept_connections(concept: str, threshold: float = 0.7):
    """Get interdisciplinary connections for a concept."""
    connections = knowledge_graph.find_interdisciplinary_connections(concept, threshold)
    return {
        "concept": concept,
        "connections": connections
    }

@app.get("/api/v1/knowledge-graph/disciplines/{discipline1}/{discipline2}")
async def get_discipline_overlap(discipline1: str, discipline2: str):
    """Get concepts that bridge two disciplines."""
    overlapping_concepts = knowledge_graph.get_discipline_overlap(discipline1, discipline2)
    return {
        "discipline1": discipline1,
        "discipline2": discipline2,
        "overlapping_concepts": overlapping_concepts
    }

@app.get("/api/v1/knowledge-graph/central-concepts")
async def get_central_concepts(top_k: int = 10):
    """Get the most central concepts in the knowledge graph."""
    central_concepts = knowledge_graph.get_central_concepts(top_k)
    return {
        "central_concepts": central_concepts
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 