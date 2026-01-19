"""
GGUF Export for Concept Predictor

Pioneering: Export MLP concept predictors to GGUF format for edge deployment.
This allows the concept predictor to run alongside quantized LLMs in llama.cpp-style runtimes.

GGUF is flexible enough to store any tensor model - we define a custom architecture
type "concept_predictor" that inference engines can recognize.
"""

import struct
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path


# GGUF Magic and Version
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF Value Types
class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

# GGML Types for tensors
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30


class GGUFWriter:
    """Write GGUF files for concept predictor models."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, np.ndarray] = {}
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value
    
    def add_tensor(self, name: str, tensor: np.ndarray):
        """Add a tensor to the model."""
        self.tensors[name] = tensor.astype(np.float32)
    
    def _write_string(self, f, s: str):
        """Write a GGUF string (length-prefixed UTF-8)."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_metadata_value(self, f, value: Any):
        """Write a metadata value with its type."""
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUFValueType.BOOL))
            f.write(struct.pack('<?', value))
        elif isinstance(value, int):
            if value >= 0:
                f.write(struct.pack('<I', GGUFValueType.UINT64))
                f.write(struct.pack('<Q', value))
            else:
                f.write(struct.pack('<I', GGUFValueType.INT64))
                f.write(struct.pack('<q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUFValueType.FLOAT32))
            f.write(struct.pack('<f', value))
        elif isinstance(value, str):
            f.write(struct.pack('<I', GGUFValueType.STRING))
            self._write_string(f, value)
        elif isinstance(value, (list, tuple)):
            f.write(struct.pack('<I', GGUFValueType.ARRAY))
            if len(value) == 0:
                f.write(struct.pack('<I', GGUFValueType.UINT32))
                f.write(struct.pack('<Q', 0))
            else:
                # Determine array type from first element
                first = value[0]
                if isinstance(first, int):
                    f.write(struct.pack('<I', GGUFValueType.UINT64))
                    f.write(struct.pack('<Q', len(value)))
                    for v in value:
                        f.write(struct.pack('<Q', v))
                elif isinstance(first, float):
                    f.write(struct.pack('<I', GGUFValueType.FLOAT32))
                    f.write(struct.pack('<Q', len(value)))
                    for v in value:
                        f.write(struct.pack('<f', v))
                elif isinstance(first, str):
                    f.write(struct.pack('<I', GGUFValueType.STRING))
                    f.write(struct.pack('<Q', len(value)))
                    for v in value:
                        self._write_string(f, v)
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")
    
    def write(self):
        """Write the GGUF file."""
        with open(self.path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))
            
            # Metadata
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_metadata_value(f, value)
            
            # Tensor info (before tensor data)
            tensor_data_offset = f.tell()
            tensor_infos = []
            
            # Calculate offsets
            current_offset = 0
            for name, tensor in self.tensors.items():
                # Align to 32 bytes
                alignment = 32
                if current_offset % alignment != 0:
                    current_offset += alignment - (current_offset % alignment)
                
                tensor_infos.append({
                    'name': name,
                    'dims': tensor.shape,
                    'type': GGMLType.F32,
                    'offset': current_offset,
                })
                current_offset += tensor.nbytes
            
            # Write tensor info
            for info in tensor_infos:
                self._write_string(f, info['name'])
                f.write(struct.pack('<I', len(info['dims'])))
                for dim in info['dims']:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', info['type']))
                f.write(struct.pack('<Q', info['offset']))
            
            # Align before tensor data
            alignment = 32
            pos = f.tell()
            if pos % alignment != 0:
                padding = alignment - (pos % alignment)
                f.write(b'\x00' * padding)
            
            tensor_data_start = f.tell()
            
            # Write tensor data
            for info in tensor_infos:
                name = info['name']
                tensor = self.tensors[name]
                
                # Align
                pos = f.tell() - tensor_data_start
                expected_offset = info['offset']
                if pos < expected_offset:
                    f.write(b'\x00' * (expected_offset - pos))
                
                f.write(tensor.tobytes())
        
        print(f"Wrote GGUF file: {self.path} ({self.path.stat().st_size / 1024:.1f} KB)")


def export_concept_predictor_gguf(
    predictor_path: str,
    output_path: str,
    text_encoder_name: str = "Alibaba-NLP/gte-large-en-v1.5",
    quantize: bool = False
) -> str:
    """
    Export a ConceptPredictor to GGUF format.
    
    Args:
        predictor_path: Path to saved predictor checkpoint (.pt)
        output_path: Output GGUF file path
        text_encoder_name: Name of text encoder used
        quantize: Whether to quantize weights (future feature)
    
    Returns:
        Path to the created GGUF file
    """
    # Load checkpoint
    checkpoint = torch.load(predictor_path, map_location='cpu')
    
    writer = GGUFWriter(output_path)
    
    # Architecture metadata
    writer.add_metadata("general.architecture", "concept_predictor")
    writer.add_metadata("general.name", "concept-first-codegen")
    writer.add_metadata("general.author", "rileyseaburg")
    writer.add_metadata("general.license", "MIT")
    writer.add_metadata("general.description", "JEPA-style concept predictor for code generation")
    
    # Model architecture
    writer.add_metadata("concept_predictor.text_encoder", text_encoder_name)
    writer.add_metadata("concept_predictor.text_dim", checkpoint.get("text_dim", 1024))
    writer.add_metadata("concept_predictor.concept_dim", checkpoint.get("concept_dim", 256))
    writer.add_metadata("concept_predictor.hidden_dim", 1024)
    writer.add_metadata("concept_predictor.num_layers", 3)
    writer.add_metadata("concept_predictor.activation", "gelu")
    
    # Quantization info
    writer.add_metadata("concept_predictor.quantization", "f32" if not quantize else "q8_0")
    
    # Add projector weights
    state_dict = checkpoint["projector_state_dict"]
    
    for name, tensor in state_dict.items():
        np_tensor = tensor.numpy()
        gguf_name = f"projector.{name}"
        writer.add_tensor(gguf_name, np_tensor)
        print(f"  Added tensor: {gguf_name} {np_tensor.shape}")
    
    writer.write()
    return output_path


def load_concept_predictor_gguf(gguf_path: str) -> Dict[str, np.ndarray]:
    """
    Load a ConceptPredictor from GGUF format.
    
    Note: This is a minimal loader. For full inference, use the
    predictor class with load_from_gguf() method.
    
    Args:
        gguf_path: Path to GGUF file
    
    Returns:
        Dictionary of tensor name -> numpy array
    """
    # This would require a full GGUF reader implementation
    # For now, return a placeholder
    raise NotImplementedError(
        "GGUF loading requires gguf-py library. "
        "Install with: pip install gguf"
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python export_gguf.py <predictor.pt> <output.gguf>")
        sys.exit(1)
    
    export_concept_predictor_gguf(sys.argv[1], sys.argv[2])
