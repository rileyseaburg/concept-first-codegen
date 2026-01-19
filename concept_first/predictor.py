"""
Concept Predictor - JEPA-style prediction from text to code concept space.

Maps natural language queries to code concept embeddings using contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os


class ConceptPredictor(nn.Module):
    """
    JEPA-style concept predictor.
    
    Given a text query, predicts the concept embedding of the target code.
    This is the key insight from VL-JEPA: predict in embedding space first,
    then decode to tokens.
    
    Architecture:
        - Text encoder (frozen): GTE-Large or similar MTEB top model
        - Projection head (trainable): MLP mapping text → code concept space
    
    Example:
        >>> predictor = ConceptPredictor(concept_dim=256)
        >>> embedding = predictor.predict("implement binary search")
        >>> print(embedding.shape)  # torch.Size([256])
    """
    
    def __init__(
        self,
        text_encoder_name: str = "Alibaba-NLP/gte-large-en-v1.5",
        concept_dim: int = 256,
        hidden_dim: int = 1024,
        device: str = None
    ):
        """
        Initialize the concept predictor.
        
        Args:
            text_encoder_name: HuggingFace model for text encoding.
                Options:
                - "Alibaba-NLP/gte-large-en-v1.5" (1024 dim, best quality)
                - "BAAI/bge-large-en-v1.5" (1024 dim, good alternative)
                - "sentence-transformers/all-mpnet-base-v2" (768 dim, faster)
            concept_dim: Output dimension (should match ConceptEncoder)
            hidden_dim: Hidden layer dimension in projection MLP
            device: Device to use
        """
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Text encoder (frozen)
        print(f"Loading text encoder: {text_encoder_name}")
        self.text_encoder = SentenceTransformer(text_encoder_name, trust_remote_code=True)
        self.text_encoder.to(self.device)
        text_dim = self.text_encoder.get_sentence_embedding_dimension()
        print(f"Text embedding dim: {text_dim}")
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Projection head (trainable)
        self.projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, concept_dim)
        ).to(self.device)
        
        self.concept_dim = concept_dim
        self.text_dim = text_dim
        
        trainable = sum(p.numel() for p in self.projector.parameters())
        print(f"Trainable parameters: {trainable:,}")
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text queries using frozen encoder."""
        with torch.no_grad():
            embeddings = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )
        # Clone to allow gradient computation through projector
        return embeddings.clone()
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Predict concept embeddings from text queries.
        
        Args:
            texts: List of natural language queries
            
        Returns:
            Normalized concept embeddings of shape (len(texts), concept_dim)
        """
        text_embeddings = self.encode_text(texts)
        concept_embeddings = self.projector(text_embeddings)
        return F.normalize(concept_embeddings, p=2, dim=-1)
    
    @torch.no_grad()
    def predict(self, query: str) -> torch.Tensor:
        """
        Predict concept embedding for a single query (inference mode).
        
        Args:
            query: Natural language description of desired code
            
        Returns:
            Normalized concept embedding of shape (concept_dim,)
        """
        self.eval()
        return self.forward([query])[0]
    
    def save(self, path: str):
        """Save the predictor (projector weights only)."""
        torch.save({
            "projector_state_dict": self.projector.state_dict(),
            "concept_dim": self.concept_dim,
            "text_dim": self.text_dim,
        }, path)
        print(f"Saved predictor to {path}")
    
    @classmethod
    def load(
        cls, 
        path: str, 
        text_encoder_name: str = "Alibaba-NLP/gte-large-en-v1.5",
        device: str = None
    ) -> "ConceptPredictor":
        """
        Load a saved predictor.
        
        Args:
            path: Path to saved checkpoint
            text_encoder_name: Text encoder to use
            device: Device to load to
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        predictor = cls(
            text_encoder_name=text_encoder_name,
            concept_dim=checkpoint["concept_dim"],
            hidden_dim=1024,  # Infer from weights if needed
            device=device
        )
        predictor.projector.load_state_dict(checkpoint["projector_state_dict"])
        predictor.eval()
        
        print(f"Loaded predictor from {path}")
        return predictor


def info_nce_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss (same as CLIP/VL-JEPA).
    
    Encourages predicted embeddings to match their corresponding targets
    while pushing away from other examples in the batch.
    
    Args:
        predicted: Predicted concept embeddings (B, D)
        target: Target code embeddings (B, D)
        temperature: Softmax temperature (lower = sharper)
        
    Returns:
        Scalar loss value
    """
    # Ensure normalized
    predicted = F.normalize(predicted, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    
    # Compute similarity matrix
    logits = (predicted @ target.T) / temperature  # (B, B)
    
    # Labels: diagonal elements are positives
    labels = torch.arange(len(predicted), device=predicted.device)
    
    # Symmetric loss (bidirectional like CLIP)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.T, labels)
    
    return (loss_p2t + loss_t2p) / 2
