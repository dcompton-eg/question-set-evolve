"""Setup script for question-set-evolve."""

from setuptools import setup, find_packages

setup(
    name="question-set-evolve",
    version="0.1.0",
    description="AI-powered interview question set and scoring rubric evolution",
    author="Your Name",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-ai>=0.1.0",
        "openai>=1.0.0",
        "matplotlib>=3.7.0",
        "python-dotenv>=1.0.0",
        "logfire>=0.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "question-evolve=question_set_evolve.cli:main",
        ],
    },
)
