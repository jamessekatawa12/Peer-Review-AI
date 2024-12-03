from typing import Dict, List, Set, Tuple
import networkx as nx
from dataclasses import dataclass
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ConceptNode:
    """Data class for storing concept information in the knowledge graph."""
    name: str
    discipline: str
    description: str
    related_terms: Set[str]
    embedding: np.ndarray

class KnowledgeGraph:
    """Knowledge graph for managing interdisciplinary concepts and relationships."""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        self.graph = nx.Graph()
        self.nlp = spacy.load("en_core_web_sm")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.concept_embeddings = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the NLP models."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text using the transformer model."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    def _extract_key_concepts(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract key concepts and their disciplines from text."""
        doc = self.nlp(text)
        concepts = []
        
        # Extract noun phrases and named entities
        for chunk in doc.noun_chunks:
            if not any(token.is_stop for token in chunk):
                concepts.append((chunk.text, self._infer_discipline(chunk.text), chunk.root.vector_norm))
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
                concepts.append((ent.text, self._infer_discipline(ent.text), ent.vector_norm))
        
        # Sort by importance (vector norm) and remove duplicates
        concepts = sorted(set(concepts), key=lambda x: x[2], reverse=True)
        return concepts[:50]  # Limit to top 50 concepts
    
    def _infer_discipline(self, concept: str) -> str:
        """Infer the discipline of a concept using keyword matching."""
        concept_lower = concept.lower()
        
        discipline_keywords = {
            "computer_science": ["algorithm", "computation", "programming", "software", "data"],
            "biology": ["cell", "organism", "gene", "protein", "evolution"],
            "physics": ["energy", "force", "quantum", "particle", "wave"],
            "chemistry": ["molecule", "reaction", "compound", "acid", "bond"],
            "mathematics": ["theorem", "proof", "equation", "matrix", "function"],
            "psychology": ["behavior", "cognitive", "perception", "memory", "emotion"],
            "sociology": ["society", "culture", "social", "community", "institution"]
        }
        
        for discipline, keywords in discipline_keywords.items():
            if any(keyword in concept_lower for keyword in keywords):
                return discipline
        
        return "interdisciplinary"
    
    def add_concept(self, name: str, discipline: str, description: str, related_terms: Set[str]):
        """Add a concept to the knowledge graph."""
        embedding = self._get_embedding(f"{name} {description}")
        
        concept = ConceptNode(
            name=name,
            discipline=discipline,
            description=description,
            related_terms=related_terms,
            embedding=embedding
        )
        
        self.graph.add_node(name, data=concept)
        self.concept_embeddings[name] = embedding
        
        # Add edges to related concepts
        for term in related_terms:
            if term in self.graph:
                similarity = cosine_similarity(
                    [self.concept_embeddings[name]],
                    [self.concept_embeddings[term]]
                )[0][0]
                self.graph.add_edge(name, term, weight=similarity)
    
    def find_interdisciplinary_connections(self, concept: str, threshold: float = 0.7) -> List[Dict]:
        """Find interdisciplinary connections for a given concept."""
        if concept not in self.graph:
            return []
        
        concept_data = self.graph.nodes[concept]["data"]
        connections = []
        
        for node in self.graph.nodes():
            if node == concept:
                continue
                
            node_data = self.graph.nodes[node]["data"]
            if node_data.discipline != concept_data.discipline:
                similarity = cosine_similarity(
                    [concept_data.embedding],
                    [node_data.embedding]
                )[0][0]
                
                if similarity >= threshold:
                    connections.append({
                        "concept": node,
                        "discipline": node_data.discipline,
                        "similarity": similarity,
                        "description": node_data.description
                    })
        
        return sorted(connections, key=lambda x: x["similarity"], reverse=True)
    
    def analyze_manuscript(self, content: str) -> Dict:
        """Analyze a manuscript for interdisciplinary connections."""
        # Extract concepts from manuscript
        concepts = self._extract_key_concepts(content)
        
        # Add new concepts to the graph
        for concept, discipline, _ in concepts:
            if concept not in self.graph:
                self.add_concept(
                    name=concept,
                    discipline=discipline,
                    description=f"Concept extracted from manuscript: {concept}",
                    related_terms=set()
                )
        
        # Find interdisciplinary connections
        connections = []
        for concept, _, _ in concepts:
            concept_connections = self.find_interdisciplinary_connections(concept)
            if concept_connections:
                connections.append({
                    "concept": concept,
                    "connections": concept_connections
                })
        
        # Calculate discipline distribution
        discipline_counts = {}
        for _, discipline, _ in concepts:
            discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
        
        return {
            "concepts": [c[0] for c in concepts],
            "interdisciplinary_connections": connections,
            "discipline_distribution": discipline_counts,
            "num_concepts": len(concepts)
        }
    
    def get_subgraph(self, concept: str, depth: int = 2) -> nx.Graph:
        """Get a subgraph centered around a concept."""
        if concept not in self.graph:
            return nx.Graph()
        
        nodes = {concept}
        current_depth = 0
        
        while current_depth < depth:
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.neighbors(node))
            nodes.update(new_nodes)
            current_depth += 1
        
        return self.graph.subgraph(nodes)
    
    def get_central_concepts(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the most central concepts in the knowledge graph."""
        centrality = nx.eigenvector_centrality(self.graph, weight="weight")
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_discipline_overlap(self, discipline1: str, discipline2: str) -> List[Dict]:
        """Find concepts that bridge two disciplines."""
        overlapping_concepts = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]["data"]
            neighbors = list(self.graph.neighbors(node))
            
            if node_data.discipline == discipline1:
                for neighbor in neighbors:
                    neighbor_data = self.graph.nodes[neighbor]["data"]
                    if neighbor_data.discipline == discipline2:
                        overlapping_concepts.append({
                            "concept1": node,
                            "concept2": neighbor,
                            "similarity": self.graph[node][neighbor]["weight"]
                        })
        
        return sorted(overlapping_concepts, key=lambda x: x["similarity"], reverse=True) 