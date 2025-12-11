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

    async def generate_question_set(
        self, candidate: EvolutionCandidate
    ) -> QuestionSetOutput:
        """Generate a question set using the candidate's prompt."""
        result = await question_writer_agent.run(candidate.question_prompt)
        return result.output

    async def generate_rubric(
        self, candidate: EvolutionCandidate
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
        return result.output

    async def evaluate_candidate(
        self, candidate: EvolutionCandidate
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
        return result.output

    async def mutate_candidate(
        self, candidate: EvolutionCandidate
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
        return result.output

    async def process_candidate(self, candidate: EvolutionCandidate) -> None:
        """Generate, evaluate, and score a candidate."""
        # Generate question set
        question_output = await self.generate_question_set(candidate)
        candidate.question_set = question_output.question_set

        # Generate rubric
        rubric_output = await self.generate_rubric(candidate)
        candidate.rubric = rubric_output.rubric

        # Evaluate
        candidate.feedback = await self.evaluate_candidate(candidate)
        candidate.score = candidate.feedback.scores.overall_average

    async def run_generation(self, generation: int) -> GenerationResult:
        """Run a single generation of evolution."""
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        # Process all candidates in parallel
        print(f"Processing {len(self.population)} candidates...")
        await asyncio.gather(
            *[self.process_candidate(c) for c in self.population]
        )

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

        result = GenerationResult(
            generation=generation,
            best_candidate=best,
            average_score=sum(scores) / len(scores),
            all_scores=scores,
        )
        self.generation_results.append(result)

        # Create next generation through tournament selection
        await self._evolve_population()

        return result

    async def _evolve_population(self) -> None:
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
                    mutated = await self.mutate_candidate(survivor)
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
                mutated = await self.mutate_candidate(survivor)
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

        for gen in range(1, generations + 1):
            await self.run_generation(gen)

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
