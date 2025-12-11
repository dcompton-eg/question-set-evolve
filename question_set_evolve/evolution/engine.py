"""Evolution engine for co-evolving question sets and scoring rubrics."""

import asyncio
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agents.question_writer import question_writer_agent, QuestionSetOutput
from ..agents.rubric_writer import rubric_writer_agent, RubricOutput, create_rubric_prompt
from ..agents.llm_as_judge import judge_agent, JudgeFeedback, create_judge_prompt
from ..agents.mutator import mutator_agent, MutatedPrompts, create_mutation_prompt
from ..models import QuestionSet, ScoringRubric


# Claude Haiku 4.5 pricing (per 1M tokens) - update these as needed
INPUT_PRICE_PER_1M = 0.80  # $ per 1M input tokens
OUTPUT_PRICE_PER_1M = 4.00  # $ per 1M output tokens


@dataclass
class TokenUsage:
    """Track token usage and costs."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Add tokens to the running total."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def merge(self, other: "TokenUsage") -> None:
        """Merge another usage tracker into this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    @property
    def input_cost(self) -> float:
        """Calculate input token cost."""
        return (self.input_tokens / 1_000_000) * INPUT_PRICE_PER_1M

    @property
    def output_cost(self) -> float:
        """Calculate output token cost."""
        return (self.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M

    @property
    def total_cost(self) -> float:
        """Calculate total cost."""
        return self.input_cost + self.output_cost

    def __str__(self) -> str:
        return (
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | "
            f"Cost: ${self.input_cost:.4f} in + ${self.output_cost:.4f} out = ${self.total_cost:.4f}"
        )


@dataclass
class EvolutionCandidate:
    """A candidate in the evolution process."""

    question_prompt: str
    rubric_prompt_additions: str = ""
    question_set: Optional[QuestionSet] = None
    rubric: Optional[ScoringRubric] = None
    feedback: Optional[JudgeFeedback] = None
    score: float = 0.0


@dataclass
class GenerationResult:
    """Results from a single generation."""

    generation: int
    best_candidate: EvolutionCandidate
    average_score: float
    all_scores: list[float] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


class QuestionSetEvolutionEngine:
    """Engine for evolving question sets and rubrics through tournament selection."""

    def __init__(
        self,
        base_question_prompt: str,
        population_size: int = 4,
        tournament_size: int = 2,
        mutation_rate: float = 0.5,
        output_dir: Optional[Path] = None,
    ):
        self.base_question_prompt = base_question_prompt
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize population with base prompt
        self.population: list[EvolutionCandidate] = [
            EvolutionCandidate(question_prompt=base_question_prompt)
            for _ in range(population_size)
        ]

        self.generation_results: list[GenerationResult] = []
        self.total_usage = TokenUsage()

    def _extract_usage(self, result) -> tuple[int, int]:
        """Extract input/output tokens from an agent result."""
        try:
            usage = result.usage()
            return usage.request_tokens or 0, usage.response_tokens or 0
        except Exception:
            # Fallback if usage not available
            return 0, 0

    async def generate_question_set(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> QuestionSetOutput:
        """Generate a question set using the candidate's prompt."""
        result = await question_writer_agent.run(candidate.question_prompt)
        input_tokens, output_tokens = self._extract_usage(result)
        usage.add(input_tokens, output_tokens)
        return result.output

    async def generate_rubric(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> RubricOutput:
        """Generate a rubric for the candidate's question set."""
        if candidate.question_set is None:
            raise ValueError("Question set must be generated first")

        prompt = create_rubric_prompt(
            candidate.question_set,
            candidate.question_prompt,
        )
        # Add any evolved rubric additions
        if candidate.rubric_prompt_additions:
            prompt += f"\n\n## Additional Requirements\n{candidate.rubric_prompt_additions}"

        result = await rubric_writer_agent.run(prompt)
        input_tokens, output_tokens = self._extract_usage(result)
        usage.add(input_tokens, output_tokens)
        return result.output

    async def evaluate_candidate(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> JudgeFeedback:
        """Evaluate a candidate's question set and rubric."""
        if candidate.question_set is None or candidate.rubric is None:
            raise ValueError("Both question set and rubric must be generated first")

        prompt = create_judge_prompt(
            candidate.question_set,
            candidate.rubric,
            candidate.question_prompt,
        )
        result = await judge_agent.run(prompt)
        input_tokens, output_tokens = self._extract_usage(result)
        usage.add(input_tokens, output_tokens)
        return result.output

    async def mutate_candidate(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> MutatedPrompts:
        """Mutate a candidate's prompts based on feedback."""
        if candidate.feedback is None:
            raise ValueError("Candidate must be evaluated first")

        prompt = create_mutation_prompt(
            candidate.question_prompt,
            candidate.rubric_prompt_additions,
            candidate.feedback,
        )
        result = await mutator_agent.run(prompt)
        input_tokens, output_tokens = self._extract_usage(result)
        usage.add(input_tokens, output_tokens)
        return result.output

    async def process_candidate(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> None:
        """Generate, evaluate, and score a candidate."""
        # Generate question set
        question_output = await self.generate_question_set(candidate, usage)
        candidate.question_set = question_output.question_set

        # Generate rubric
        rubric_output = await self.generate_rubric(candidate, usage)
        candidate.rubric = rubric_output.rubric

        # Evaluate
        candidate.feedback = await self.evaluate_candidate(candidate, usage)
        candidate.score = candidate.feedback.scores.overall_average

    async def run_generation(self, generation: int) -> GenerationResult:
        """Run a single generation of evolution."""
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        # Track usage for this generation
        gen_usage = TokenUsage()

        # Process all candidates in parallel
        print(f"Processing {len(self.population)} candidates...")

        # Create per-candidate usage trackers for parallel execution
        candidate_usages = [TokenUsage() for _ in self.population]
        await asyncio.gather(
            *[
                self.process_candidate(c, u)
                for c, u in zip(self.population, candidate_usages)
            ]
        )

        # Merge all candidate usages
        for u in candidate_usages:
            gen_usage.merge(u)

        # Sort by score
        self.population.sort(key=lambda c: c.score, reverse=True)

        # Report scores
        scores = [c.score for c in self.population]
        print(f"Scores: {[f'{s:.1f}' for s in scores]}")
        print(f"Average: {sum(scores) / len(scores):.1f}")
        print(f"Best: {scores[0]:.1f}")

        # Save best candidate
        best = self.population[0]
        self._save_generation(generation, best)

        # Create next generation through tournament selection
        mutation_usage = TokenUsage()
        await self._evolve_population(mutation_usage)
        gen_usage.merge(mutation_usage)

        # Update total usage
        self.total_usage.merge(gen_usage)

        # Print usage for this generation
        print(f"\nGeneration {generation} usage: {gen_usage}")
        print(f"Total usage so far: {self.total_usage}")

        result = GenerationResult(
            generation=generation,
            best_candidate=best,
            average_score=sum(scores) / len(scores),
            all_scores=scores,
            usage=gen_usage,
        )
        self.generation_results.append(result)

        return result

    async def _evolve_population(self, usage: TokenUsage) -> None:
        """Evolve the population through selection and mutation."""
        # Keep top performers
        survivors = self.population[: self.tournament_size]

        # Always keep best unchanged
        new_population = [
            EvolutionCandidate(
                question_prompt=survivors[0].question_prompt,
                rubric_prompt_additions=survivors[0].rubric_prompt_additions,
            )
        ]

        # Mutate survivors and fill population
        for survivor in survivors:
            if len(new_population) >= self.population_size:
                break

            if random.random() < self.mutation_rate:
                # Mutate this survivor
                try:
                    mutated = await self.mutate_candidate(survivor, usage)
                    new_population.append(
                        EvolutionCandidate(
                            question_prompt=mutated.question_prompt,
                            rubric_prompt_additions=mutated.rubric_prompt_additions,
                        )
                    )
                    print(f"Mutated candidate: {mutated.mutation_rationale[:100]}...")
                except Exception as e:
                    print(f"Mutation failed: {e}, keeping original")
                    new_population.append(
                        EvolutionCandidate(
                            question_prompt=survivor.question_prompt,
                            rubric_prompt_additions=survivor.rubric_prompt_additions,
                        )
                    )
            else:
                # Keep unchanged
                new_population.append(
                    EvolutionCandidate(
                        question_prompt=survivor.question_prompt,
                        rubric_prompt_additions=survivor.rubric_prompt_additions,
                    )
                )

        # Fill remaining slots with mutations of random survivors
        while len(new_population) < self.population_size:
            survivor = random.choice(survivors)
            try:
                mutated = await self.mutate_candidate(survivor, usage)
                new_population.append(
                    EvolutionCandidate(
                        question_prompt=mutated.question_prompt,
                        rubric_prompt_additions=mutated.rubric_prompt_additions,
                    )
                )
            except Exception as e:
                print(f"Mutation failed: {e}, keeping original")
                new_population.append(
                    EvolutionCandidate(
                        question_prompt=survivor.question_prompt,
                        rubric_prompt_additions=survivor.rubric_prompt_additions,
                    )
                )

        self.population = new_population

    def _save_generation(self, generation: int, candidate: EvolutionCandidate) -> None:
        """Save the best candidate from a generation."""
        # Save question prompt
        prompt_path = self.output_dir / f"generation_{generation}_question_prompt.txt"
        prompt_path.write_text(candidate.question_prompt)

        # Save rubric additions
        if candidate.rubric_prompt_additions:
            rubric_prompt_path = (
                self.output_dir / f"generation_{generation}_rubric_additions.txt"
            )
            rubric_prompt_path.write_text(candidate.rubric_prompt_additions)

        # Save question set as JSON
        if candidate.question_set:
            qs_path = self.output_dir / f"generation_{generation}_questions.json"
            qs_path.write_text(candidate.question_set.model_dump_json(indent=2))

        # Save rubric as JSON
        if candidate.rubric:
            rubric_path = self.output_dir / f"generation_{generation}_rubric.json"
            rubric_path.write_text(candidate.rubric.model_dump_json(indent=2))

        # Save feedback
        if candidate.feedback:
            feedback_path = self.output_dir / f"generation_{generation}_feedback.json"
            feedback_path.write_text(candidate.feedback.model_dump_json(indent=2))

        print(f"Saved generation {generation} to {self.output_dir}")

    async def evolve(self, generations: int) -> list[GenerationResult]:
        """Run the full evolution process."""
        print(f"Starting evolution with {generations} generations")
        print(f"Population size: {self.population_size}")
        print(f"Tournament size: {self.tournament_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Pricing: ${INPUT_PRICE_PER_1M}/1M input, ${OUTPUT_PRICE_PER_1M}/1M output")

        for gen in range(1, generations + 1):
            await self.run_generation(gen)

        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL TOKEN USAGE SUMMARY")
        print(f"{'='*60}")
        print(f"Total input tokens:  {self.total_usage.input_tokens:,}")
        print(f"Total output tokens: {self.total_usage.output_tokens:,}")
        print(f"Total input cost:    ${self.total_usage.input_cost:.4f}")
        print(f"Total output cost:   ${self.total_usage.output_cost:.4f}")
        print(f"TOTAL COST:          ${self.total_usage.total_cost:.4f}")

        # Plot results if matplotlib available
        self._plot_results()

        return self.generation_results

    def _plot_results(self) -> None:
        """Plot evolution progress."""
        try:
            import matplotlib.pyplot as plt

            generations = [r.generation for r in self.generation_results]
            avg_scores = [r.average_score for r in self.generation_results]
            best_scores = [r.best_candidate.score for r in self.generation_results]

            plt.figure(figsize=(10, 6))
            plt.plot(generations, avg_scores, "b-o", label="Average Score")
            plt.plot(generations, best_scores, "g-^", label="Best Score")
            plt.xlabel("Generation")
            plt.ylabel("Score")
            plt.title("Question Set Evolution Progress")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "evolution_progress.png", dpi=150)
            plt.close()
            print(f"Saved evolution plot to {self.output_dir / 'evolution_progress.png'}")
        except ImportError:
            print("matplotlib not available, skipping plot")
