"""
Concept Encoder - Encodes code into semantic embeddings.

Uses CodeT5+ embedding model for state-of-the-art code representations.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from tqdm.auto import tqdm


class ConceptEncoder:
    """
    Encodes code snippets into semantic concept embeddings.
    
    Uses CodeT5+ 110M embedding model - specifically trained for code embeddings.
    This captures semantic similarity between code patterns.
    
    Example:
        >>> encoder = ConceptEncoder()
        >>> embedding = encoder.encode("def fibonacci(n): ...")
        >>> print(embedding.shape)  # torch.Size([256])
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/codet5p-110m-embedding",
        device: str = None
    ):
        """
        Initialize the concept encoder.
        
        Args:
            model_name: HuggingFace model ID for code embeddings.
                Options:
                - "Salesforce/codet5p-110m-embedding" (default, best quality)
                - "microsoft/unixcoder-base" (alternative)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        print(f"Loading concept encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Determine embedding dimension
        with torch.no_grad():
            test_input = self.tokenizer("def test(): pass", return_tensors="pt").to(self.device)
            test_output = self.model(**test_input)
            if hasattr(test_output, 'last_hidden_state'):
                self.embed_dim = test_output.last_hidden_state.shape[-1]
            else:
                self.embed_dim = test_output.shape[-1]
        
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @torch.no_grad()
    def encode(self, code: str) -> torch.Tensor:
        """
        Encode a single code snippet into a normalized embedding.
        
        Args:
            code: Python code string
            
        Returns:
            Normalized embedding tensor of shape (embed_dim,)
        """
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        # Handle different model output formats
        if hasattr(outputs, 'last_hidden_state'):
            # Mean pooling over sequence
            mask = inputs["attention_mask"].unsqueeze(-1)
            embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # Model returns embeddings directly
            embedding = outputs
            if embedding.dim() == 3:
                embedding = embedding.mean(dim=1)
        
        # L2 normalize
        return F.normalize(embedding.squeeze(0), p=2, dim=-1)
    
    @torch.no_grad()
    def encode_batch(
        self, 
        codes: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode multiple code snippets.
        
        Args:
            codes: List of Python code strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Normalized embeddings tensor of shape (len(codes), embed_dim)
        """
        all_embeddings = []
        
        iterator = range(0, len(codes), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", leave=False)
        
        for i in iterator:
            batch = codes[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            if hasattr(outputs, 'last_hidden_state'):
                mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                embeddings = outputs
                if embeddings.dim() == 3:
                    embeddings = embeddings.mean(dim=1)
            
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
