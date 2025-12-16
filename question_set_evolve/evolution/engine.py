"""Evolution engine for co-evolving question sets and scoring rubrics."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agents.question_writer import question_writer_agent
from ..agents.rubric_writer import rubric_writer_agent, create_rubric_prompt
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
            f"Cost: ${
                self.input_cost:.4f} in + ${self.output_cost:.4f} out = ${self.total_cost:.4f}"
        )


@dataclass
class EvolutionCandidate:
    """A candidate in the evolution process."""

    id: str  # Unique identifier (e.g., "P0", "C1", "C2")
    question_prompt: str
    # ID of parent candidate (None for initial)
    parent_id: Optional[str] = None
    rubric_prompt_additions: str = ""
    # Explanation of mutations applied to create this candidate
    mutation_rationale: str = ""
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
    """Engine for evolving question sets and rubrics using (1+λ) Evolution Strategy.

    The (1+λ) ES is the standard algorithm for "high-cost" optimization:
    - 1 Parent: The current best candidate (champion)
    - λ Children: Mutants generated from the parent each generation

    Each generation:
    1. Generate λ mutants from the current best
    2. Evaluate all children
    3. If any child beats the parent, it becomes the new parent
    4. Otherwise, the parent survives unchanged
    """

    def __init__(
        self,
        base_question_prompt: str,
        num_children: int = 4,
        output_dir: Optional[Path] = None,
    ):
        self.base_question_prompt = base_question_prompt
        self.num_children = num_children  # λ in (1+λ) ES
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ID counter for generating unique candidate IDs
        self._next_id = 0

        # Initialize with a single parent (the champion)
        self.parent = EvolutionCandidate(
            id=self._generate_id("P"),
            question_prompt=base_question_prompt,
        )

        self.generation_results: list[GenerationResult] = []
        self.total_usage = TokenUsage()

    def _generate_id(self, prefix: str = "C") -> str:
        """Generate a unique ID for a candidate.

        Args:
            prefix: "P" for parent/promoted, "C" for child
        """
        candidate_id = f"{prefix}{self._next_id}"
        self._next_id += 1
        return candidate_id

    def _extract_usage(self, result) -> tuple[int, int]:
        """Extract input/output tokens from an agent result."""
        try:
            usage = result.usage()
            return usage.input_tokens or 0, usage.output_tokens or 0
        except Exception:
            # Fallback if usage not available
            return 0, 0

    async def generate_question_set(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> QuestionSet:
        """Generate a question set using the candidate's prompt."""
        result = await question_writer_agent.run(candidate.question_prompt)
        input_tokens, output_tokens = self._extract_usage(result)
        usage.add(input_tokens, output_tokens)
        return result.output.question_set

    async def generate_rubric(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> ScoringRubric:
        """Generate a rubric for the candidate's question set."""
        if candidate.question_set is None:
            raise ValueError("Question set must be generated first")

        prompt = create_rubric_prompt(
            candidate.question_set,
            candidate.question_prompt,
        )
        # Add any evolved rubric additions
        if candidate.rubric_prompt_additions:
            prompt += f"\n\n## Additional Requirements\n{
                candidate.rubric_prompt_additions}"

        result = await rubric_writer_agent.run(prompt)
        input_tokens, output_tokens = self._extract_usage(result)
        usage.add(input_tokens, output_tokens)
        return result.output.rubric

    async def evaluate_candidate(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> JudgeFeedback:
        """Evaluate a candidate's question set and rubric."""
        if candidate.question_set is None or candidate.rubric is None:
            raise ValueError(
                "Both question set and rubric must be generated first")

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

        # TODO: there may be a blind spot in that the base prompt is not included in mutation.
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
        """Generate, evaluate, and score a candidate.

        If any step fails, the candidate gets a score of 0 and processing continues.
        """
        try:
            # Generate question set
            candidate.question_set = await self.generate_question_set(candidate, usage)
        except Exception as e:
            print(f"  Failed to generate question set: {e}")
            candidate.score = 0.0
            return

        try:
            # Generate rubric
            candidate.rubric = await self.generate_rubric(candidate, usage)
        except Exception as e:
            print(f"  Failed to generate rubric: {e}")
            candidate.score = 0.0
            return

        try:
            # Evaluate
            candidate.feedback = await self.evaluate_candidate(candidate, usage)
            candidate.score = candidate.feedback.scores.overall_average
        except Exception as e:
            print(f"  Failed to evaluate candidate: {e}")
            candidate.score = 0.0

    async def run_generation(self, generation: int) -> GenerationResult:
        """Run a single generation of (1+λ) evolution.

        Algorithm:
        1. If first generation, evaluate the parent
        2. Generate λ children by mutating the parent
        3. Evaluate all children in parallel
        4. If best child > parent, child becomes new parent
        5. Otherwise, parent survives unchanged
        """
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        gen_usage = TokenUsage()

        # First generation: evaluate the initial parent
        if generation == 1:
            print(f"\n[INIT] Evaluating initial parent [{self.parent.id}]...")
            await self.process_candidate(self.parent, gen_usage)
            print(f"[INIT] Parent [{self.parent.id}] score: {
                  self.parent.score:.1f}")

        # Show current parent status
        print(f"\n[PARENT] Current champion: [{self.parent.id}] score={
              self.parent.score:.1f}")
        if self.parent.parent_id:
            print(f"         Lineage: {
                  self.parent.parent_id} -> {self.parent.id}")

        # Generate λ children by mutating the parent
        print(f"\n[MUTATE] Generating {
              self.num_children} children from parent [{self.parent.id}]...")
        children = await self._generate_children(gen_usage)

        # Print mutation details for each child
        for child in children:
            if child.mutation_rationale:
                print(f"\n[{child.id}] Mutation rationale:")
                # Print each line indented
                for line in child.mutation_rationale.strip().split("\n"):
                    print(f"         {line}")

        # Process all children in parallel
        print(f"\n[EVAL] Evaluating {len(children)} children...")
        child_usages = [TokenUsage() for _ in children]
        await asyncio.gather(
            *[
                self.process_candidate(c, u)
                for c, u in zip(children, child_usages)
            ]
        )

        # Merge all child usages
        for u in child_usages:
            gen_usage.merge(u)

        # Find best child
        children.sort(key=lambda c: c.score, reverse=True)
        best_child = children[0]

        # Report all scores with IDs
        print(f"\n[SCORES] Generation {generation} results:")
        print(f"         Parent [{self.parent.id}]: {self.parent.score:.1f}")
        for child in children:
            marker = " <-- BEST CHILD" if child.id == best_child.id else ""
            print(f"         Child  [{child.id}] (from {
                  child.parent_id}): {child.score:.1f}{marker}")

        # Selection: compare best child against parent
        all_scores = [self.parent.score] + [c.score for c in children]
        print(f"\n[SELECT] Comparing best child [{
              best_child.id}] vs parent [{self.parent.id}]...")
        if best_child.score > self.parent.score:
            print(f"         ✓ Child [{best_child.id}] beats parent ({
                  best_child.score:.1f} > {self.parent.score:.1f})")
            print(f"         [{best_child.id}] is promoted to new parent!")
            self.parent = best_child
        else:
            print(f"         ✗ Parent [{self.parent.id}] survives (no child beat {
                  self.parent.score:.1f})")

        # Save best (current parent)
        self._save_generation(generation, self.parent)

        # Update total usage
        self.total_usage.merge(gen_usage)

        # Print usage for this generation
        print(f"\n[USAGE] Generation {generation}: {gen_usage}")
        print(f"[USAGE] Total so far: {self.total_usage}")

        result = GenerationResult(
            generation=generation,
            best_candidate=self.parent,
            average_score=sum(all_scores) / len(all_scores),
            all_scores=all_scores,
            usage=gen_usage,
        )
        self.generation_results.append(result)

        return result

    async def _generate_children(self, usage: TokenUsage) -> list[EvolutionCandidate]:
        """Generate λ children by mutating the current parent.

        Each child is an independent mutation of the parent.
        If mutation fails, we retry; if it fails repeatedly, we skip that child.
        """
        children = []

        # Generate children in parallel using mutation
        mutation_tasks = []
        mutation_usages = [TokenUsage() for _ in range(self.num_children)]

        for i in range(self.num_children):
            mutation_tasks.append(
                self._try_mutate_candidate(self.parent, mutation_usages[i])
            )

        results = await asyncio.gather(*mutation_tasks, return_exceptions=True)

        # Collect successful mutations
        for i, result in enumerate(results):
            usage.merge(mutation_usages[i])
            if isinstance(result, Exception):
                print(f"         Mutation {i+1} failed: {result}")
            elif result is not None:
                mutated = result
                child_id = self._generate_id("C")
                child = EvolutionCandidate(
                    id=child_id,
                    question_prompt=mutated.question_prompt,
                    parent_id=self.parent.id,
                    rubric_prompt_additions=mutated.rubric_prompt_additions,
                    mutation_rationale=mutated.mutation_rationale,
                )
                children.append(child)
                print(f"         [{child_id}] from [{self.parent.id}]")

        if not children:
            # Fallback: if all mutations failed, create one clone (shouldn't happen often)
            clone_id = self._generate_id("C")
            print(
                f"         Warning: All mutations failed, creating clone [{clone_id}]")
            children.append(
                EvolutionCandidate(
                    id=clone_id,
                    question_prompt=self.parent.question_prompt,
                    parent_id=self.parent.id,
                    rubric_prompt_additions=self.parent.rubric_prompt_additions,
                )
            )

        return children

    async def _try_mutate_candidate(
        self, candidate: EvolutionCandidate, usage: TokenUsage
    ) -> Optional[MutatedPrompts]:
        """Try to mutate a candidate, returning None if it fails."""
        try:
            return await self.mutate_candidate(candidate, usage)
        except Exception as e:
            return None

    def _save_generation(self, generation: int, candidate: EvolutionCandidate) -> None:
        """Save the best candidate from a generation."""
        # Save question prompt
        prompt_path = self.output_dir / \
            f"generation_{generation}_question_prompt.txt"
        prompt_path.write_text(candidate.question_prompt)

        # Save rubric additions
        if candidate.rubric_prompt_additions:
            rubric_prompt_path = (
                self.output_dir /
                f"generation_{generation}_rubric_additions.txt"
            )
            rubric_prompt_path.write_text(candidate.rubric_prompt_additions)

        # Save mutation rationale
        if candidate.mutation_rationale:
            rationale_path = (
                self.output_dir /
                f"generation_{generation}_mutation_rationale.txt"
            )
            rationale_path.write_text(candidate.mutation_rationale)

        # Save question set as JSON
        if candidate.question_set:
            qs_path = self.output_dir / \
                f"generation_{generation}_questions.json"
            qs_path.write_text(
                candidate.question_set.model_dump_json(indent=2))

        # Save rubric as JSON
        if candidate.rubric:
            rubric_path = self.output_dir / \
                f"generation_{generation}_rubric.json"
            rubric_path.write_text(candidate.rubric.model_dump_json(indent=2))

        # Save feedback
        if candidate.feedback:
            feedback_path = self.output_dir / \
                f"generation_{generation}_feedback.json"
            feedback_path.write_text(
                candidate.feedback.model_dump_json(indent=2))

        print(f"Saved generation {generation} to {self.output_dir}")

    async def evolve(self, generations: int) -> list[GenerationResult]:
        """Run the full (1+λ) evolution process."""
        print(f"Starting (1+λ) Evolution Strategy")
        print(f"Generations: {generations}")
        print(f"Children per generation (λ): {self.num_children}")
        print(f"Initial parent: [{self.parent.id}]")
        print(f"Pricing: ${
              INPUT_PRICE_PER_1M}/1M input, ${OUTPUT_PRICE_PER_1M}/1M output")

        for gen in range(1, generations + 1):
            await self.run_generation(gen)

        # Print final summary
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")

        # Build lineage chain
        lineage = [self.parent.id]
        current_parent_id = self.parent.parent_id
        while current_parent_id:
            lineage.insert(0, current_parent_id)
            # Find the parent in generation results to continue the chain
            found = False
            for result in self.generation_results:
                if result.best_candidate.id == current_parent_id:
                    current_parent_id = result.best_candidate.parent_id
                    found = True
                    break
            if not found:
                break

        print(f"\n[WINNER] Final champion: [{self.parent.id}] score={
              self.parent.score:.1f}")
        print(f"[LINEAGE] {' -> '.join(lineage)}")

        # Score progression
        print(f"\n[HISTORY] Score progression:")
        for result in self.generation_results:
            print(f"         Gen {result.generation}: [{result.best_candidate.id}] = {
                  result.best_candidate.score:.1f}")

        print(f"\n{'='*60}")
        print("TOKEN USAGE SUMMARY")
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
            best_scores = [
                r.best_candidate.score for r in self.generation_results]

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
            print(f"Saved evolution plot to {
                  self.output_dir / 'evolution_progress.png'}")
        except ImportError:
            print("matplotlib not available, skipping plot")
