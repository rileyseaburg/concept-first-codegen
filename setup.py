"""Setup script for concept-first-codegen."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="concept-first-codegen",
    version="0.1.0",
    author="Riley Seaburg",
    author_email="riley@seaburg.dev",
    description="Concept-First Code Generation inspired by VL-JEPA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rileyseaburg/concept-first-codegen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "sentence-transformers>=3.0.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "bitsandbytes>=0.44.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "scikit-learn>=1.3.0",
        ],
    },
)
