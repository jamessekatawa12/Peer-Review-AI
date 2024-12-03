from typing import Any, Dict, List, Tuple
import re
import requests
from dataclasses import dataclass
from ..base.agent import BaseAgent, AgentType, ReviewResult

@dataclass
class Citation:
    """Data class for storing citation information."""
    text: str
    authors: List[str]
    year: str
    title: str
    venue: str
    doi: str = ""

class CitationVerificationAgent(BaseAgent):
    """Agent specialized in verifying and analyzing citations."""
    
    def __init__(self, name: str = "Citation Verifier"):
        super().__init__(name, AgentType.DISCIPLINARY)
        self.citation_patterns = {
            "apa": r'\b([A-Za-z-]+(?:(?:,\s*[A-Za-z-]+)+)?(?:\s*et\s*al\.)?)\s*\((\d{4})\)',
            "ieee": r'\[(\d+)\]',
            "harvard": r'\b([A-Za-z-]+(?:(?:,\s*[A-Za-z-]+)+)?(?:\s*et\s*al\.)?),\s*(\d{4})\b'
        }
        self.reference_patterns = {
            "doi": r'doi\.org/([^\s]+)',
            "url": r'https?://[^\s]+'
        }
    
    async def initialize(self) -> None:
        """Initialize the citation verification system."""
        self.is_initialized = True
    
    def _extract_citations(self, content: str) -> List[Tuple[str, str]]:
        """Extract citations from the text using various citation styles."""
        citations = []
        
        for style, pattern in self.citation_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                if style == "ieee":
                    citations.append(("ieee", match.group(1)))
                else:
                    citations.append((style, f"{match.group(1)} ({match.group(2)})"))
        
        return citations
    
    def _extract_references(self, content: str) -> List[Citation]:
        """Extract references from the bibliography section."""
        references = []
        
        # Find bibliography section
        sections = re.split(r'\n(?=[A-Z][a-z]+ *\n|[0-9]+\. [A-Z])', content)
        bibliography = ""
        
        for section in sections:
            if any(keyword in section.lower() for keyword in ["references", "bibliography", "works cited"]):
                bibliography = section
                break
        
        if not bibliography:
            return references
        
        # Split into individual references
        ref_entries = re.split(r'\n(?=\[\d+\]|\d+\.|\b[A-Za-z])', bibliography)
        
        for entry in ref_entries:
            if not entry.strip():
                continue
            
            # Extract DOI if present
            doi_match = re.search(self.reference_patterns["doi"], entry)
            doi = doi_match.group(1) if doi_match else ""
            
            # Basic parsing of reference components
            parts = entry.split(".")
            if len(parts) >= 2:
                authors_part = parts[0]
                title_part = parts[1] if len(parts) > 1 else ""
                venue_part = parts[2] if len(parts) > 2 else ""
                
                # Extract year
                year_match = re.search(r'\((\d{4})\)', entry)
                year = year_match.group(1) if year_match else ""
                
                # Extract authors
                authors = [a.strip() for a in authors_part.split(",") if a.strip()]
                
                references.append(Citation(
                    text=entry.strip(),
                    authors=authors,
                    year=year,
                    title=title_part.strip(),
                    venue=venue_part.strip(),
                    doi=doi
                ))
        
        return references
    
    def _verify_citation_consistency(self, citations: List[Tuple[str, str]], references: List[Citation]) -> List[str]:
        """Verify consistency between citations and references."""
        issues = []
        
        # Create sets of citation keys
        citation_keys = set()
        for _, cite in citations:
            # Extract author names and years from citations
            if "(" in cite:
                author, year = cite.split("(")
                year = year.rstrip(")")
                citation_keys.add(f"{author.strip()}{year}")
        
        # Create set of reference keys
        reference_keys = set()
        for ref in references:
            if ref.authors and ref.year:
                key = f"{ref.authors[0]}{ref.year}"
                reference_keys.add(key)
        
        # Find missing citations/references
        missing_refs = citation_keys - reference_keys
        uncited_refs = reference_keys - citation_keys
        
        if missing_refs:
            issues.append(f"Citations without corresponding references: {', '.join(missing_refs)}")
        if uncited_refs:
            issues.append(f"References not cited in text: {', '.join(uncited_refs)}")
        
        return issues
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        """Perform citation verification and analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        # Extract citations and references
        citations = self._extract_citations(content)
        references = self._extract_references(content)
        
        # Verify consistency
        consistency_issues = self._verify_citation_consistency(citations, references)
        
        # Calculate citation metrics
        num_citations = len(citations)
        num_references = len(references)
        citation_density = num_citations / (len(content.split()) / 1000)  # Citations per 1000 words
        
        # Generate recommendations
        recommendations = []
        if consistency_issues:
            recommendations.extend(consistency_issues)
        
        if citation_density < 1.0:
            recommendations.append("Consider adding more citations to support your claims")
        
        if num_citations == 0:
            recommendations.append("No citations found in the text")
        
        if num_references == 0:
            recommendations.append("No references found in the bibliography")
        
        # Calculate overall score
        score = 1.0
        if consistency_issues:
            score -= 0.2 * len(consistency_issues)
        if citation_density < 1.0:
            score -= 0.1
        if num_citations == 0 or num_references == 0:
            score -= 0.3
        
        score = max(0.0, score)
        
        return ReviewResult(
            score=score,
            comments=[f"Found {num_citations} citations and {num_references} references"],
            suggestions=recommendations,
            confidence=0.9,
            metadata={
                "num_citations": num_citations,
                "num_references": num_references,
                "citation_density": citation_density,
                "consistency_issues": consistency_issues
            }
        )
    
    async def collaborate(self, other_agent: BaseAgent, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents by sharing citation analysis."""
        if not isinstance(shared_data.get("content"), str):
            return {"error": "No content provided for collaboration"}
        
        review_result = await self.review(shared_data["content"], {})
        return {
            "citation_score": review_result.score,
            "citation_metrics": {
                "num_citations": review_result.metadata["num_citations"],
                "num_references": review_result.metadata["num_references"],
                "citation_density": review_result.metadata["citation_density"]
            },
            "consistency_issues": review_result.metadata["consistency_issues"]
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "Citation extraction and verification",
            "Reference list analysis",
            "Citation-reference consistency checking",
            "Citation density analysis",
            "Multiple citation style support",
            "DOI verification",
            "Bibliography formatting analysis"
        ] 