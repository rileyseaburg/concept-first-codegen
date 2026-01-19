#!/usr/bin/env python3
"""
Basic usage example for Concept-First Code Generation.

This script demonstrates how to use the library to generate code
with concept-guided retrieval.
"""

from concept_first import (
    ConceptEncoder,
    ConceptPredictor, 
    ConceptBank,
    ConceptFirstGenerator
)


def main():
    # Initialize components
    print("=" * 60)
    print("Concept-First Code Generation - Basic Usage")
    print("=" * 60)
    
    # 1. Load the concept encoder
    print("\n1. Loading Concept Encoder...")
    encoder = ConceptEncoder()
    
    # 2. Load or train the concept predictor
    print("\n2. Loading Concept Predictor...")
    try:
        predictor = ConceptPredictor.load(
            "checkpoints/concept_predictor.pt",
            device="cuda"
        )
    except FileNotFoundError:
        print("   No checkpoint found, initializing new predictor...")
        predictor = ConceptPredictor(concept_dim=encoder.embed_dim)
    
    # 3. Load or build the concept bank
    print("\n3. Loading Concept Bank...")
    try:
        bank = ConceptBank.load("checkpoints/concept_bank.pt", encoder)
    except FileNotFoundError:
        print("   No checkpoint found, creating empty bank...")
        bank = ConceptBank(encoder)
        
        # Add some example code
        example_codes = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "def is_palindrome(s):\n    return s == s[::-1]",
        ]
        example_descs = [
            "Calculate the nth Fibonacci number recursively",
            "Calculate factorial of n recursively",
            "Check if a string is a palindrome",
        ]
        bank.add(example_codes, example_descs, source="examples")
    
    # 4. Create the generator
    print("\n4. Initializing Generator...")
    generator = ConceptFirstGenerator(
        concept_predictor=predictor,
        concept_bank=bank,
        llm_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",  # Use smaller model for demo
        num_examples=2,
        use_4bit=True
    )
    
    # 5. Generate code!
    print("\n5. Generating Code...")
    print("=" * 60)
    
    queries = [
        "write a function to check if a number is prime",
        "implement binary search on a sorted list",
        "reverse a linked list"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = generator.generate(query, show_examples=True)
        
        print(f"\nGenerated Code:")
        print(result["code"][:500])
        print("-" * 40)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
