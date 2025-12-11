#!/bin/bash
# Generate a question set and scoring rubric

set -e

PYTHON=python3
PROMPT_FILE="${1:-prompts/golang_backend.txt}"
OUTPUT_DIR="${2:-output}"

echo "Question Set Generator"
echo "======================"
echo "Prompt: $PROMPT_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY environment variable not set"
  echo "Run: export ANTHROPIC_API_KEY=your-key"
  exit 1
fi

# Check prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: Prompt file not found: $PROMPT_FILE"
  exit 1
fi

# Activate venv if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Run generation
$PYTHON -m question_set_evolve.cli generate \
  --prompt-file "$PROMPT_FILE" \
  --output "$OUTPUT_DIR"

echo ""
echo "Done! Output files:"
echo "  - $OUTPUT_DIR/questions.json"
echo "  - $OUTPUT_DIR/rubric.json"
