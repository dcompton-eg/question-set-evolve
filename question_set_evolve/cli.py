"""Command-line interface for question set evolution."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .evolution.engine import QuestionSetEvolutionEngine, TokenUsage, INPUT_PRICE_PER_1M, OUTPUT_PRICE_PER_1M
from .agents.question_writer import question_writer_agent
from .agents.rubric_writer import rubric_writer_agent, create_rubric_prompt
from .agents.candidate_scorer import candidate_scorer_agent, create_scoring_prompt
from .models import QuestionSet, ScoringRubric

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
    prompt = load_prompt(args.prompt, args.prompt_file)

    output_dir = Path(args.output) if args.output else Path("output")

    engine = QuestionSetEvolutionEngine(
        base_question_prompt=prompt,
        population_size=args.population,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
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
        "--population", "-p", type=int, default=4, help="Population size"
    )
    evolve_parser.add_argument(
        "--tournament-size", "-t", type=int, default=2, help="Tournament size"
    )
    evolve_parser.add_argument(
        "--mutation-rate", "-m", type=float, default=0.5, help="Mutation rate"
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

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(cmd_generate(args))
    elif args.command == "evolve":
        asyncio.run(cmd_evolve(args))
    elif args.command == "score":
        asyncio.run(cmd_score(args))


if __name__ == "__main__":
    main()
