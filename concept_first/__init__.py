"""
Concept-First Code Generation

Inspired by VL-JEPA: Predict concept embeddings first, then generate code.
"""

from .encoder import ConceptEncoder
from .predictor import ConceptPredictor
from .bank import ConceptBank
from .generator import ConceptFirstGenerator

__version__ = "0.1.0"
__all__ = ["ConceptEncoder", "ConceptPredictor", "ConceptBank", "ConceptFirstGenerator"]
