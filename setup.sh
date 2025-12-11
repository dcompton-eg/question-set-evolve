#!/bin/bash
# Setup script for question-set-evolve

set -e

echo "Setting up question-set-evolve..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing package..."
pip install -e .

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  source venv/bin/activate"
echo ""
echo "  # Generate a single question set and rubric"
echo "  question-evolve generate --prompt-file prompt.txt"
echo ""
echo "  # Run evolution for 5 generations"
echo "  question-evolve evolve --prompt-file prompt.txt --generations 5"
echo ""
echo "  # Score a candidate transcript"
echo "  question-evolve score --questions output/questions.json --rubric output/rubric.json --transcript transcript.txt"
echo ""
echo "Make sure to set your OPENAI_API_KEY environment variable!"
