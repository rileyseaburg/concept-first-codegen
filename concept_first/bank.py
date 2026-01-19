"""
Concept Bank - Searchable database of code concepts.

Stores code embeddings and enables semantic retrieval.
"""

import torch
from typing import List, Dict, Optional
from collections import Counter

from .encoder import ConceptEncoder


class ConceptBank:
    """
    A searchable bank of code concepts.
    
    Maps embeddings to code snippets for retrieval. This serves as the
    "memory" of code patterns that guides generation.
    
    Example:
        >>> bank = ConceptBank(encoder)
        >>> bank.add(codes, descriptions, source="mbpp")
        >>> results = bank.search_by_text("fibonacci recursive")
        >>> print(results[0]["code"])
    """
    
    def __init__(self, encoder: ConceptEncoder):
        """
        Initialize the concept bank.
        
        Args:
            encoder: ConceptEncoder instance for encoding new code
        """
        self.encoder = encoder
        self.embeddings: Optional[torch.Tensor] = None  # (N, embed_dim)
        self.codes: List[str] = []
        self.descriptions: List[str] = []
        self.sources: List[str] = []
    
    def add(
        self, 
        codes: List[str], 
        descriptions: List[str] = None, 
        source: str = "unknown"
    ):
        """
        Add code snippets to the bank.
        
        Args:
            codes: List of code strings
            descriptions: List of descriptions (docstrings, comments, etc.)
            source: Source identifier (e.g., "mbpp", "humaneval")
        """
        if descriptions is None:
            descriptions = [""] * len(codes)
        
        # Filter out empty/invalid codes
        valid_pairs = [
            (c, d) for c, d in zip(codes, descriptions) 
            if c and len(c.strip()) > 10
        ]
        
        if not valid_pairs:
            print(f"  No valid codes from {source}")
            return
        
        codes, descriptions = zip(*valid_pairs)
        codes, descriptions = list(codes), list(descriptions)
        
        print(f"  Encoding {len(codes)} examples from {source}...")
        new_embeddings = self.encoder.encode_batch(codes)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
        
        self.codes.extend(codes)
        self.descriptions.extend(descriptions)
        self.sources.extend([source] * len(codes))
        
        print(f"  Bank size: {len(self.codes)} concepts")
    
    def search(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict]:
        """
        Find k nearest concepts to the query embedding.
        
        Args:
            query_embedding: Query embedding from predictor or encoder
            k: Number of results to return
            
        Returns:
            List of dicts with keys: code, description, similarity, source
        """
        if self.embeddings is None or len(self.codes) == 0:
            return []
        
        query_embedding = query_embedding.cpu()
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Compute similarities
        similarities = (query_embedding @ self.embeddings.T).squeeze(0)
        top_k = similarities.topk(min(k, len(self.codes)))
        
        results = []
        for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
            results.append({
                "code": self.codes[idx],
                "description": self.descriptions[idx],
                "similarity": score,
                "source": self.sources[idx]
            })
        
        return results
    
    def search_by_code(self, code: str, k: int = 5) -> List[Dict]:
        """
        Find similar code snippets.
        
        Args:
            code: Query code string
            k: Number of results
            
        Returns:
            List of similar code entries
        """
        embedding = self.encoder.encode(code)
        return self.search(embedding, k)
    
    def search_by_text(self, text: str, k: int = 5) -> List[Dict]:
        """
        Find code matching a text description.
        
        Note: This encodes text as if it were code. For better results,
        use a ConceptPredictor to map text to concept space first.
        
        Args:
            text: Natural language query
            k: Number of results
            
        Returns:
            List of matching code entries
        """
        embedding = self.encoder.encode(text)
        return self.search(embedding, k)
    
    def stats(self):
        """Print statistics about the concept bank."""
        source_counts = Counter(self.sources)
        print(f"\nConcept Bank Statistics:")
        print(f"  Total concepts: {len(self.codes)}")
        if self.embeddings is not None:
            print(f"  Embedding dim: {self.embeddings.shape[1]}")
        print(f"  Sources:")
        for source, count in source_counts.most_common():
            print(f"    - {source}: {count}")
    
    def save(self, path: str):
        """
        Save the concept bank to disk.
        
        Args:
            path: Output file path (.pt)
        """
        torch.save({
            "embeddings": self.embeddings,
            "codes": self.codes,
            "descriptions": self.descriptions,
            "sources": self.sources
        }, path)
        print(f"Saved concept bank to {path}")
    
    @classmethod
    def load(cls, path: str, encoder: ConceptEncoder) -> "ConceptBank":
        """
        Load a saved concept bank.
        
        Args:
            path: Path to saved checkpoint
            encoder: ConceptEncoder instance (for future additions)
            
        Returns:
            Loaded ConceptBank
        """
        data = torch.load(path, map_location="cpu")
        
        bank = cls(encoder)
        bank.embeddings = data["embeddings"]
        bank.codes = data["codes"]
        bank.descriptions = data["descriptions"]
        bank.sources = data.get("sources", ["unknown"] * len(data["codes"]))
        
        print(f"Loaded concept bank from {path}")
        bank.stats()
        
        return bank
