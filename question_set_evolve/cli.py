"""Command-line interface for question set evolution."""

import argparse
import asyncio
import json
import logging
import re
import shutil
from pathlib import Path

from .models import QuestionSet, ScoringRubric, JudgeFeedback

# Set up logging - only show warnings and errors
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)


def load_prompt(prompt_arg: str | None, prompt_file_arg: str | None) -> str:
    """Load prompt from argument or file."""
    if prompt_arg:
        return prompt_arg
    if prompt_file_arg:
        return Path(prompt_file_arg).read_text()
    raise ValueError("Must provide either --prompt or --prompt-file")


def extract_usage(result) -> tuple[int, int]:
    """Extract input/output tokens from an agent result."""
    try:
        usage = result.usage()
        return usage.input_tokens or 0, usage.output_tokens or 0
    except Exception:
        return 0, 0


async def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a single question set and rubric."""
    from .evolution.engine import TokenUsage
    from .agents.question_writer import question_writer_agent
    from .agents.rubric_writer import rubric_writer_agent, create_rubric_prompt

    prompt = load_prompt(args.prompt, args.prompt_file)
    total_usage = TokenUsage()

    print("=" * 60)
    print("STEP 1: Generating Question Set")
    print("=" * 60)
    question_result = await question_writer_agent.run(prompt)
    input_tokens, output_tokens = extract_usage(question_result)
    total_usage.add(input_tokens, output_tokens)
    print(f"Tokens: {input_tokens:,} in / {output_tokens:,} out")

    question_set = question_result.output.question_set
    print(f"\nQuestion Set: {question_set.title}")
    print(f"Target Role: {question_set.target_role}")
    print(f"Total Time: {question_set.total_time_minutes} minutes")
    print(f"Description: {question_set.description[:150]}...")
    print(f"\nQuestions ({len(question_set.questions)}):")

    for i, q in enumerate(question_set.questions, 1):
        print(f"\n  Q{i} [{q.category}] [{q.difficulty}] [{q.time_allocation_minutes}min]")
        print(f"  {q.question_text[:200]}{'...' if len(q.question_text) > 200 else ''}")

    # Generate rubric
    print("\n" + "=" * 60)
    print("STEP 2: Generating Scoring Rubric")
    print("=" * 60)
    rubric_prompt = create_rubric_prompt(question_set, prompt)
    rubric_result = await rubric_writer_agent.run(rubric_prompt)
    input_tokens, output_tokens = extract_usage(rubric_result)
    total_usage.add(input_tokens, output_tokens)
    print(f"Tokens: {input_tokens:,} in / {output_tokens:,} out")

    rubric = rubric_result.output.rubric
    print(f"\nRubric: {rubric.title}")
    print(f"Overall Guidance: {rubric.overall_scoring_guidance[:200]}...")
    print(f"\nQuestion Rubrics ({len(rubric.question_rubrics)}):")
    for qr in rubric.question_rubrics:
        print(f"\n  [{qr.question_id}]")
        print(f"  Criteria: {len(qr.criteria)}")
        for c in qr.criteria:
            print(f"    - {c.name} (weight: {c.weight})")
        if qr.red_flags:
            print(f"  Red flags: {len(qr.red_flags)}")
        if qr.bonus_indicators:
            print(f"  Bonus indicators: {len(qr.bonus_indicators)}")

    # Save outputs
    print("\n" + "=" * 60)
    print("STEP 3: Saving Outputs")
    print("=" * 60)

    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    qs_path = output_dir / "questions.json"
    qs_path.write_text(question_set.model_dump_json(indent=2))
    print(f"Saved questions to {qs_path}")

    rubric_path = output_dir / "rubric.json"
    rubric_path.write_text(rubric.model_dump_json(indent=2))
    print(f"Saved rubric to {rubric_path}")

    # Print usage summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input tokens:  {total_usage.input_tokens:,}")
    print(f"Output tokens: {total_usage.output_tokens:,}")
    print(f"Total cost:    ${total_usage.total_cost:.4f}")
    print("\nDone!")


async def cmd_evolve(args: argparse.Namespace) -> None:
    """Run the evolution process."""
    from .evolution.engine import QuestionSetEvolutionEngine

    prompt = load_prompt(args.prompt, args.prompt_file)

    output_dir = Path(args.output) if args.output else Path("output")

    engine = QuestionSetEvolutionEngine(
        base_question_prompt=prompt,
        num_children=args.children,
        output_dir=output_dir,
    )

    results = await engine.evolve(args.generations)

    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print("=" * 60)

    print(f"\nGenerations: {len(results)}")
    print(f"Final best score: {results[-1].best_candidate.score:.1f}")
    print(f"Score improvement: {results[-1].best_candidate.score - results[0].best_candidate.score:+.1f}")
    print(f"\nOutputs saved to: {output_dir}")


async def cmd_score(args: argparse.Namespace) -> None:
    """Score a candidate transcript using a rubric."""
    from .agents.candidate_scorer import candidate_scorer_agent, create_scoring_prompt

    # Load rubric
    rubric_path = Path(args.rubric)
    rubric_data = json.loads(rubric_path.read_text())
    rubric = ScoringRubric.model_validate(rubric_data)

    # Load questions
    questions_path = Path(args.questions)
    questions_data = json.loads(questions_path.read_text())
    question_set = QuestionSet.model_validate(questions_data)

    # Load transcript
    transcript_path = Path(args.transcript)
    transcript = transcript_path.read_text()

    print(f"Scoring candidate transcript...")
    print(f"Questions: {questions_path}")
    print(f"Rubric: {rubric_path}")
    print(f"Transcript: {transcript_path}")

    # Score the candidate
    scoring_prompt = create_scoring_prompt(question_set, rubric, transcript)
    result = await candidate_scorer_agent.run(scoring_prompt)
    evaluation = result.output

    print(f"\n{'='*60}")
    print("CANDIDATE EVALUATION")
    print("=" * 60)
    print(f"\nOverall Score: {evaluation.overall_score:.2f}/5.0")
    print(f"Recommendation: {evaluation.hiring_recommendation}")

    print("\nQuestion Scores:")
    for qs in evaluation.question_scores:
        print(f"  {qs.question_id}: {qs.overall_question_score:.2f}/5.0")

    print("\nStrengths:")
    for s in evaluation.strengths:
        print(f"  - {s}")

    print("\nAreas for Improvement:")
    for a in evaluation.areas_for_improvement:
        print(f"  - {a}")

    print(f"\nSummary:\n{evaluation.summary}")

    # Save evaluation
    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = output_dir / f"evaluation_{args.candidate_id}.json"
    eval_path.write_text(evaluation.model_dump_json(indent=2))
    print(f"\nSaved evaluation to {eval_path}")


def cmd_select_best(args: argparse.Namespace) -> None:
    """Select the best generation and create best_ files with PDF."""
    from .pdf_generator import generate_question_set_pdf

    output_dir = Path(args.output) if args.output else Path("output")

    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' does not exist")
        return

    # Find all generation feedback files
    feedback_pattern = re.compile(r"generation_(\d+)_feedback\.json")
    generations: list[tuple[int, float, Path]] = []

    for feedback_file in output_dir.glob("generation_*_feedback.json"):
        match = feedback_pattern.match(feedback_file.name)
        if match:
            gen_num = int(match.group(1))
            try:
                feedback_data = json.loads(feedback_file.read_text())
                feedback = JudgeFeedback.model_validate(feedback_data)
                score = feedback.scores.overall_average
                generations.append((gen_num, score, feedback_file))
            except Exception as e:
                print(f"Warning: Could not parse {feedback_file}: {e}")

    if not generations:
        print("No generation feedback files found in output directory")
        return

    # Sort by score descending
    generations.sort(key=lambda x: x[1], reverse=True)

    print(f"{'='*60}")
    print("Generation Scores")
    print(f"{'='*60}")
    for gen_num, score, _ in generations:
        marker = " <-- BEST" if gen_num == generations[0][0] else ""
        print(f"  Generation {gen_num}: {score:.1f}{marker}")

    best_gen, best_score, _ = generations[0]
    print(f"\nBest generation: {best_gen} (score: {best_score:.1f})")

    # Copy best files
    print(f"\n{'='*60}")
    print("Copying Best Files")
    print(f"{'='*60}")

    questions_src = output_dir / f"generation_{best_gen}_questions.json"
    rubric_src = output_dir / f"generation_{best_gen}_rubric.json"
    prompt_src = output_dir / f"generation_{best_gen}_question_prompt.txt"

    questions_dst = output_dir / "best_questions.json"
    rubric_dst = output_dir / "best_rubric.json"
    prompt_dst = output_dir / "best_question_prompt.txt"

    if questions_src.exists():
        shutil.copy(questions_src, questions_dst)
        print(f"Copied {questions_src.name} -> {questions_dst.name}")
    else:
        print(f"Warning: {questions_src.name} not found")

    if rubric_src.exists():
        shutil.copy(rubric_src, rubric_dst)
        print(f"Copied {rubric_src.name} -> {rubric_dst.name}")
    else:
        print(f"Warning: {rubric_src.name} not found")

    if prompt_src.exists():
        shutil.copy(prompt_src, prompt_dst)
        print(f"Copied {prompt_src.name} -> {prompt_dst.name}")

    # Generate PDF
    print(f"\n{'='*60}")
    print("Generating PDF")
    print(f"{'='*60}")

    if questions_dst.exists():
        try:
            questions_data = json.loads(questions_dst.read_text())
            question_set = QuestionSet.model_validate(questions_data)

            pdf_path = output_dir / "best_questions.pdf"
            generate_question_set_pdf(question_set, pdf_path)
            print(f"Generated PDF: {pdf_path}")
        except Exception as e:
            print(f"Error generating PDF: {e}")
    else:
        print("Cannot generate PDF: best_questions.json not found")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  {questions_dst}")
    print(f"  {rubric_dst}")
    if (output_dir / "best_questions.pdf").exists():
        print(f"  {output_dir / 'best_questions.pdf'}")


def cmd_pdf(args: argparse.Namespace) -> None:
    """Generate a PDF from a question set JSON file."""
    from .pdf_generator import generate_question_set_pdf

    output_dir = Path(args.output) if args.output else Path("output")

    # Determine the questions file to use
    if args.questions:
        # Direct path to questions file
        questions_path = Path(args.questions)
        if not questions_path.exists():
            print(f"Error: Questions file '{questions_path}' not found")
            return
        # Default PDF output name based on input file
        default_pdf_name = questions_path.stem + ".pdf"
    elif args.generation is not None:
        # Generation number specified
        questions_path = output_dir / f"generation_{args.generation}_questions.json"
        if not questions_path.exists():
            print(f"Error: Generation {args.generation} questions file not found at '{questions_path}'")
            return
        default_pdf_name = f"generation_{args.generation}_questions.pdf"
    else:
        print("Error: Must specify either --generation or --questions")
        return

    # Determine output PDF path
    if args.pdf_output:
        pdf_path = Path(args.pdf_output)
    else:
        pdf_path = output_dir / default_pdf_name

    # Ensure output directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and validate questions
    print(f"{'='*60}")
    print("Generating PDF")
    print(f"{'='*60}")
    print(f"Source: {questions_path}")

    try:
        questions_data = json.loads(questions_path.read_text())
        question_set = QuestionSet.model_validate(questions_data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{questions_path}': {e}")
        return
    except Exception as e:
        print(f"Error: Could not parse question set: {e}")
        return

    # Generate PDF
    try:
        generate_question_set_pdf(question_set, pdf_path)
        print(f"Generated: {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"\nQuestion Set: {question_set.title}")
    print(f"Questions: {len(question_set.questions)}")
    print(f"PDF: {pdf_path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Question Set Evolve - AI-powered interview question and rubric evolution"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate", help="Generate a single question set and rubric"
    )
    gen_parser.add_argument("--prompt", help="The prompt describing the question set")
    gen_parser.add_argument("--prompt-file", help="File containing the prompt")
    gen_parser.add_argument("--output", "-o", help="Output directory", default="output")

    # Evolve command
    evolve_parser = subparsers.add_parser(
        "evolve", help="Run the evolution process"
    )
    evolve_parser.add_argument("--prompt", help="The base prompt for evolution")
    evolve_parser.add_argument("--prompt-file", help="File containing the base prompt")
    evolve_parser.add_argument("--output", "-o", help="Output directory", default="output")
    evolve_parser.add_argument(
        "--generations", "-g", type=int, default=5, help="Number of generations"
    )
    evolve_parser.add_argument(
        "--children", "-c", type=int, default=4,
        help="Number of children (mutants) per generation (λ in (1+λ) ES)"
    )

    # Score command
    score_parser = subparsers.add_parser(
        "score", help="Score a candidate transcript"
    )
    score_parser.add_argument("--questions", "-q", required=True, help="Questions JSON file")
    score_parser.add_argument("--rubric", "-r", required=True, help="Rubric JSON file")
    score_parser.add_argument("--transcript", "-t", required=True, help="Candidate transcript file")
    score_parser.add_argument("--candidate-id", "-c", default="candidate", help="Candidate identifier")
    score_parser.add_argument("--output", "-o", help="Output directory", default="output")

    # Select-best command
    select_best_parser = subparsers.add_parser(
        "select-best", help="Select the best generation and create best_ files with PDF"
    )
    select_best_parser.add_argument("--output", "-o", help="Output directory", default="output")

    # PDF command
    pdf_parser = subparsers.add_parser(
        "pdf", help="Generate a PDF from a question set"
    )
    pdf_parser.add_argument(
        "--generation", "-g", type=int, help="Generation number to generate PDF for"
    )
    pdf_parser.add_argument(
        "--questions", "-q", help="Path to questions JSON file (alternative to --generation)"
    )
    pdf_parser.add_argument("--output", "-o", help="Output directory", default="output")
    pdf_parser.add_argument(
        "--pdf-output", "-p", help="Output PDF file path (overrides default naming)"
    )

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(cmd_generate(args))
    elif args.command == "evolve":
        asyncio.run(cmd_evolve(args))
    elif args.command == "score":
        asyncio.run(cmd_score(args))
    elif args.command == "select-best":
        cmd_select_best(args)
    elif args.command == "pdf":
        cmd_pdf(args)


if __name__ == "__main__":
    main()
