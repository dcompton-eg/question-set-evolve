# Question Set Evolve

An AI-powered system for automatically generating, evaluating, and iteratively improving interview question sets and scoring rubrics using (1+λ) Evolution Strategy with LLM-as-judge feedback.

## Use Case

Creating effective interview materials is time-consuming and often inconsistent. Question Set Evolve automates the entire interview preparation and evaluation pipeline:

1. **Generate tailored interview questions** - Define your requirements (e.g., "Senior Go backend engineer with 5+ years experience") and get a complete question set with follow-ups, difficulty levels, and time allocations
2. **Create aligned scoring rubrics** - Automatically generate evaluation criteria with 1-5 scales, clear anchors, red flags, and example answers
3. **Evolve and improve** - Use (1+λ) Evolution Strategy with LLM feedback to iteratively improve question quality across generations
4. **Score candidate transcripts** - After conducting interviews, score transcripts against the rubrics to get objective evaluations and hiring recommendations

This reduces manual effort while maintaining rigor through automated evaluation and iterative improvement.

## Features

- **Question Set Generation** - AI-powered creation of structured interview questions with difficulty levels, time allocations, and follow-up questions
- **Scoring Rubric Creation** - Automatically generates evaluation criteria aligned with questions, including 1-5 scales with clear anchors
- **Quality Evaluation** - LLM-as-Judge evaluates 11 quality dimensions across clarity, relevance, fairness, objectivity, and more
- **Iterative Evolution** - (1+λ) Evolution Strategy with LLM-driven mutation to improve prompts over generations
- **Candidate Transcript Scoring** - Score interview transcripts against rubrics with per-question scores, strengths/weaknesses analysis, and hiring recommendations (Strong Hire/Hire/Lean No Hire/No Hire)
- **PDF Generation** - Export question sets to formatted PDFs for printing or sharing
- **Cost Monitoring** - Tracks token usage and costs throughout the process

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd question-set-evolve

# Setup using the provided script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

Create a `.env` file with your API keys:

```bash
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here  # optional
LOGFIRE_TOKEN=your-logfire-token         # optional
```

## Usage

### Generate a Question Set

Create a single question set and rubric from a prompt file:

```bash
question-evolve generate --prompt-file prompt.txt --output output
```

### Evolve Question Sets

Run multi-generation evolution to improve question quality using (1+λ) Evolution Strategy:

```bash
question-evolve evolve --prompt-file prompt.txt \
  --generations 5 \
  --children 4 \
  --verbose
```

Options:
- `--generations, -g` - Number of generations to run (default: 5)
- `--children, -c` - Number of children (λ) per generation (default: 4)
- `--verbose, -v` - Print detailed output including mutation rationales

The (1+λ) ES algorithm:
1. Starts with one parent (the champion)
2. Each generation, creates λ mutant children from the parent
3. Evaluates all children against 11 quality dimensions
4. If any child beats the parent, it becomes the new champion
5. Otherwise, the parent survives unchanged

### Score a Candidate Transcript

After conducting an interview, evaluate the candidate by scoring their transcript against the generated rubric:

```bash
question-evolve score \
  --questions output/questions.json \
  --rubric output/rubric.json \
  --transcript transcript.txt \
  --candidate-id candidate_1
```

This produces:
- Per-question scores (1-5 scale)
- Overall competency score
- Identified strengths and areas for improvement
- Hiring recommendation (Strong Hire / Hire / Lean No Hire / No Hire)
- Narrative summary of the candidate's performance

### Select Best Generation

After running evolution, select the best-scoring generation and create output files:

```bash
question-evolve select-best --output output
```

This will:
- Scan all generation feedback files and rank by score
- Copy the best generation's files to `best_questions.json`, `best_rubric.json`, and `best_question_prompt.txt`
- Generate a formatted PDF at `best_questions.pdf`

### Generate PDF

Generate a PDF from any question set:

```bash
# From a specific generation
question-evolve pdf --generation 3 --output output

# From any questions JSON file
question-evolve pdf --questions path/to/questions.json

# With custom output path
question-evolve pdf --generation 3 --pdf-output my_questions.pdf
```

## Architecture

The project follows a modular, agent-based architecture using (1+λ) Evolution Strategy:

```
          ┌─────────────────────────────────────────┐
          │           For each child (λ):          │
          │  Mutate → Generate → Rubric → Evaluate │
          └─────────────────────────────────────────┘
                            ↓
                   Select best → Next Generation
```

### Core Components

| Component | Description |
|-----------|-------------|
| Evolution Engine | Central orchestrator implementing (1+λ) ES with pipelined evaluation |
| Question Writer Agent | Generates interview question sets from prompts |
| Rubric Writer Agent | Creates scoring rubrics matched to questions |
| LLM-as-Judge Agent | Evaluates quality across 11 dimensions |
| Mutator Agent | Evolves prompts based on judge feedback |
| Candidate Scorer Agent | Scores interview transcripts using rubrics |
| PDF Generator | Creates formatted PDFs from question sets |

### Quality Dimensions Evaluated

**Question Set:**
- Clarity, Relevance, Depth, Coverage, Fairness

**Rubric:**
- Objectivity, Discrimination, Calibration, LLM Compatibility

**Co-evolution:**
- Alignment, Completeness

## Project Structure

```
question_set_evolve/
├── agents/              # Specialized LLM agents
│   ├── __init__.py
│   ├── question_writer.py
│   ├── rubric_writer.py
│   ├── llm_as_judge.py
│   ├── mutator.py
│   └── candidate_scorer.py
├── evolution/           # Evolution engine
│   ├── __init__.py
│   └── engine.py
├── models.py            # Pydantic data models
├── pdf_generator.py     # PDF output generation
└── cli.py               # Command-line interface
prompts/                 # Example prompts
├── golang_backend.txt
└── react_native.txt
setup.py                 # Package configuration
```

## Requirements

- Python 3.11+
- Anthropic API key (Claude)

## Dependencies

- pydantic (v2.0+) - Data validation
- pydantic-ai (v0.1+) - LLM agent framework
- anthropic (v0.39+) - Claude API client
- openai (v1.0+) - OpenAI API support (optional)
- matplotlib (v3.7+) - Evolution progress visualization
- reportlab (v4.0+) - PDF generation
- python-dotenv (v1.0+) - Environment variable loading
- logfire (v0.30+) - Observability (optional)

## Output

The system saves all artifacts:

- Generated questions and rubrics (JSON)
- Evolution history per generation
- Judge feedback and scores
- Candidate evaluations
- Evolution progress plots (PNG)
- Best generation files (`best_questions.json`, `best_rubric.json`)
- Formatted question set PDF (`best_questions.pdf`)

## License

See LICENSE file for details.
