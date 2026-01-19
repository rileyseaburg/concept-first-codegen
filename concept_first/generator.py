"""
Concept-First Code Generator - The main pipeline.

Combines concept prediction, retrieval, and LLM generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional

from .predictor import ConceptPredictor
from .bank import ConceptBank


class ConceptFirstGenerator:
    """
    Concept-First Code Generation Pipeline.
    
    This is the main class that combines all components:
    1. Predict concept embedding from query (JEPA-style)
    2. Retrieve similar code examples from concept bank
    3. Generate code conditioned on retrieved examples
    
    Example:
        >>> generator = ConceptFirstGenerator(predictor, bank)
        >>> result = generator.generate("implement binary search tree")
        >>> print(result["code"])
    """
    
    def __init__(
        self,
        concept_predictor: ConceptPredictor,
        concept_bank: ConceptBank,
        llm_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        num_examples: int = 3,
        use_4bit: bool = True,
        device: str = None
    ):
        """
        Initialize the generator.
        
        Args:
            concept_predictor: Trained ConceptPredictor
            concept_bank: ConceptBank with code examples
            llm_name: HuggingFace model for code generation.
                Options:
                - "Qwen/Qwen3-Coder-30B-A3B-Instruct" (best, MoE 3B active)
                - "Qwen/Qwen2.5-Coder-7B-Instruct" (good quality)
                - "Qwen/Qwen2.5-Coder-3B-Instruct" (faster)
                - "Qwen/Qwen2.5-Coder-1.5B-Instruct" (fastest)
            num_examples: Number of examples to retrieve
            use_4bit: Whether to use 4-bit quantization
            device: Device to use
        """
        self.concept_predictor = concept_predictor
        self.concept_bank = concept_bank
        self.num_examples = num_examples
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load LLM
        print(f"Loading LLM: {llm_name}")
        
        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if not torch.cuda.is_available():
                self.llm = self.llm.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"LLM loaded successfully")
    
    def retrieve_examples(self, query: str) -> List[Dict]:
        """
        Retrieve relevant code examples using predicted concept.
        
        Args:
            query: Natural language query
            
        Returns:
            List of relevant code examples
        """
        concept_embedding = self.concept_predictor.predict(query)
        examples = self.concept_bank.search(concept_embedding, k=self.num_examples)
        return examples
    
    def build_prompt(self, query: str, examples: List[Dict]) -> str:
        """
        Build few-shot prompt with retrieved examples.
        
        Args:
            query: User's query
            examples: Retrieved code examples
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are an expert Python programmer. Write clean, efficient, and well-documented code.",
            "Output ONLY the code, no explanations.",
            "",
            "Here are some similar code examples for reference:"
        ]
        
        for i, ex in enumerate(examples):
            desc = ex['description'][:150] if ex['description'] else "(utility function)"
            code = ex['code'][:500]  # Truncate long code
            prompt_parts.append(f"\n### Example {i+1}: {desc}")
            prompt_parts.append(f"```python\n{code}\n```")
        
        prompt_parts.extend([
            "",
            f"### Task: {query}",
            "",
            "Write the Python code:",
            "```python"
        ])
        
        return "\n".join(prompt_parts)
    
    def extract_code(self, generated: str) -> str:
        """
        Extract Python code from model output.
        
        Args:
            generated: Raw model output
            
        Returns:
            Extracted code string
        """
        # Look for code blocks
        if "```python" in generated:
            parts = generated.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        
        if "```" in generated:
            code = generated.split("```")[0].strip()
            if code:
                return code
        
        # Look for def/class statements
        lines = generated.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'import ', 'from ', '@')):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return generated.strip()
    
    def generate(
        self,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        show_examples: bool = False
    ) -> Dict:
        """
        Generate code using concept-first approach.
        
        Args:
            query: Natural language description of desired code
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            show_examples: Whether to print retrieved examples
            
        Returns:
            Dict with keys:
                - code: Generated code string
                - examples: Retrieved examples
                - concept_similarities: Similarity scores
                - raw_output: Raw model output
        """
        # Step 1: Retrieve examples using predicted concept
        examples = self.retrieve_examples(query)
        
        if show_examples:
            print("Retrieved concept-matched examples:")
            for i, ex in enumerate(examples):
                desc = ex['description'][:40].replace('\n', ' ') if ex['description'] else "(no desc)"
                print(f"  [{i+1}] (sim={ex['similarity']:.3f}, src={ex['source']}) {desc}...")
        
        # Step 2: Build prompt
        prompt = self.build_prompt(query, examples)
        
        # Step 3: Generate with LLM
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        raw_output = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Step 4: Extract code
        code = self.extract_code(raw_output)
        
        return {
            "code": code,
            "examples": examples,
            "concept_similarities": [ex["similarity"] for ex in examples],
            "raw_output": raw_output
        }
    
    def generate_direct(self, query: str, max_new_tokens: int = 512) -> str:
        """
        Generate code directly without concept guidance (for comparison).
        
        Args:
            query: Natural language query
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated code string
        """
        prompt = f"""You are an expert Python programmer. Write clean, efficient, and well-documented code.
Output ONLY the code, no explanations.

### Task: {query}

Write the Python code:
```python"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return self.extract_code(generated)
