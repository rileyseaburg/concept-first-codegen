"""
Concept-First Code Generation with AirLLM

Run Qwen3-Coder-30B on 4GB VRAM using AirLLM's layer-wise loading.

AirLLM: https://github.com/lyogavin/airllm
- Loads model one layer at a time
- 70B models on 4GB VRAM
- 405B models on 8GB VRAM
- No quantization required (but supports 4bit/8bit for speed)

Install:
    pip install airllm

Usage:
    python examples/airllm_low_vram.py
"""

import torch
from airllm import AutoModel as AirLLMAutoModel
from concept_first import ConceptPredictor, ConceptBank, ConceptEncoder


def create_airllm_generator(
    predictor: ConceptPredictor,
    bank: ConceptBank,
    llm_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    compression: str = "4bit",  # None, "4bit", or "8bit"
):
    """
    Create a concept-first generator using AirLLM for low VRAM inference.
    
    Args:
        predictor: Trained ConceptPredictor
        bank: ConceptBank with code examples
        llm_name: Model to load (can be 70B+ models!)
        compression: "4bit" or "8bit" for faster inference, None for full precision
    
    Returns:
        Generator function
    """
    print(f"Loading {llm_name} with AirLLM (compression={compression})")
    print("This may take a while on first run as it splits the model into layers...")
    
    # Load with AirLLM - handles layer-wise loading automatically
    model = AirLLMAutoModel.from_pretrained(
        llm_name,
        compression=compression,  # 4bit gives ~3x speedup
    )
    
    print(f"Model loaded! Ready for inference on low VRAM.")
    
    def generate(
        query: str,
        num_examples: int = 3,
        max_new_tokens: int = 512,
    ) -> dict:
        """Generate code using concept-first approach with AirLLM."""
        
        # Step 1: Predict concept and retrieve examples
        concept_embedding = predictor.predict(query)
        examples = bank.search(concept_embedding, k=num_examples)
        
        print(f"Retrieved {len(examples)} concept-matched examples")
        for i, ex in enumerate(examples):
            desc = ex['description'][:40] if ex['description'] else "(no desc)"
            print(f"  [{i+1}] sim={ex['similarity']:.3f}: {desc}...")
        
        # Step 2: Build prompt with examples
        prompt_parts = [
            "You are an expert Python programmer. Write clean, efficient code.",
            "Output ONLY the code, no explanations.",
            "",
            "Here are similar code examples for reference:"
        ]
        
        for i, ex in enumerate(examples):
            desc = ex['description'][:150] if ex['description'] else "(utility)"
            code = ex['code'][:400]
            prompt_parts.append(f"\n### Example {i+1}: {desc}")
            prompt_parts.append(f"```python\n{code}\n```")
        
        prompt_parts.extend([
            "",
            f"### Task: {query}",
            "",
            "Write the Python code:",
            "```python"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Step 3: Generate with AirLLM
        input_tokens = model.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=4096,
            padding=False
        )
        
        generation_output = model.generate(
            input_tokens['input_ids'].cuda(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True
        )
        
        raw_output = model.tokenizer.decode(generation_output.sequences[0])
        
        # Extract code from output
        code = extract_code(raw_output, prompt)
        
        return {
            "code": code,
            "examples": examples,
            "raw_output": raw_output
        }
    
    return generate


def extract_code(output: str, prompt: str) -> str:
    """Extract generated code from model output."""
    # Remove the prompt prefix if present
    if prompt in output:
        output = output[len(prompt):]
    
    # Look for code blocks
    if "```python" in output:
        parts = output.split("```python")
        if len(parts) > 1:
            code = parts[-1].split("```")[0].strip()
            return code
    
    if "```" in output:
        code = output.split("```")[0].strip()
        if code:
            return code
    
    # Look for function/class definitions
    lines = output.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith(('def ', 'class ', 'import ', 'from ', '@')):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else output.strip()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Concept-First Code Generation with AirLLM")
    print("Running Qwen3-Coder-30B on low VRAM!")
    print("=" * 60)
    
    # Check VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {vram_gb:.1f} GB")
        
        if vram_gb < 8:
            print("Low VRAM detected - AirLLM is perfect for this!")
    else:
        print("\nNo GPU detected. AirLLM can also run on CPU (slower).")
    
    # Initialize components
    print("\n1. Loading Concept Encoder...")
    encoder = ConceptEncoder()
    
    print("\n2. Loading Concept Bank...")
    # You would load a pre-trained bank here
    # bank = ConceptBank.load("concept_bank.pt", encoder)
    
    print("\n3. Loading Concept Predictor...")
    # predictor = ConceptPredictor.load("concept_predictor.pt")
    
    print("\n4. Creating AirLLM generator...")
    # generate = create_airllm_generator(
    #     predictor=predictor,
    #     bank=bank,
    #     llm_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    #     compression="4bit"  # 3x faster inference
    # )
    
    print("\n5. Generating code...")
    # result = generate("implement a binary search tree with insert and delete")
    # print(result["code"])
    
    print("\n" + "=" * 60)
    print("To run this example:")
    print("1. pip install airllm")
    print("2. Train or download concept_predictor.pt and concept_bank.pt")
    print("3. Uncomment the code above")
    print("=" * 60)
