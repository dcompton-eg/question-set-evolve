"""Mutator Agent - co-evolves question and rubric prompts based on judge feedback."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .llm_as_judge import JudgeFeedback


class MutatedPrompts(BaseModel):
    """Output from the mutator agent."""

    question_prompt: str = Field(
        description="The mutated prompt for generating questions"
    )
    rubric_prompt_additions: str = Field(
        description="Additional guidance to add to the rubric prompt"
    )
    mutation_rationale: str = Field(
        description="Explanation of what was changed and why"
    )


MUTATOR_SYSTEM_PROMPT = """You are an expert at improving interview design prompts based on
evaluation feedback. Your role is to mutate prompts to address weaknesses while preserving strengths.

## Mutation Principles

### Preservation
- Keep instructions that led to high scores
- Maintain core requirements and constraints
- Preserve successful structural elements

### Improvement
- Add specific instructions to address low-scoring dimensions
- Clarify ambiguous requirements
- Add examples when helpful
- Strengthen weak areas without over-constraining

### Co-evolution
- When questions are weak, improve question prompt
- When rubric is weak, add rubric guidance
- When alignment is weak, add cross-referencing instructions
- Ensure changes to one don't break the other

## Mutation Strategies

1. **For low clarity**: Add examples of well-phrased questions
2. **For low relevance**: Add more specific role requirements
3. **For low depth**: Request layered questions with follow-ups
4. **For low coverage**: List specific skill areas to cover
5. **For low fairness**: Add bias-checking instructions

6. **For low rubric objectivity**: Request more concrete criteria
7. **For low discrimination**: Request clearer score anchors
8. **For low calibration**: Request better example answers
9. **For low LLM compatibility**: Request transcript-matchable criteria

10. **For low alignment**: Add instructions to cross-reference
11. **For low completeness**: Require rubric for every question

## Output Format
- The question prompt should be complete and self-contained
- Rubric prompt additions will be appended to the base rubric prompt
- Explain your mutations clearly so they can be evaluated

Be surgical. Make the minimum changes needed to address feedback.
"""


mutator_agent = Agent(
    "openai:gpt-4.1",
    system_prompt=MUTATOR_SYSTEM_PROMPT,
    output_type=MutatedPrompts,
    model_settings={"temperature": 0.8},
)


def create_mutation_prompt(
    original_question_prompt: str,
    current_rubric_additions: str,
    feedback: JudgeFeedback,
) -> str:
    """Create a prompt for mutating the design prompts."""
    scores = feedback.scores

    # Identify lowest scoring dimensions
    dimension_scores = [
        ("question_clarity", scores.question_clarity),
        ("question_relevance", scores.question_relevance),
        ("question_depth", scores.question_depth),
        ("question_coverage", scores.question_coverage),
        ("question_fairness", scores.question_fairness),
        ("rubric_objectivity", scores.rubric_objectivity),
        ("rubric_discrimination", scores.rubric_discrimination),
        ("rubric_calibration", scores.rubric_calibration),
        ("rubric_llm_compatibility", scores.rubric_llm_compatibility),
        ("alignment", scores.alignment),
        ("completeness", scores.completeness),
    ]
    sorted_dims = sorted(dimension_scores, key=lambda x: x[1])
    weakest = sorted_dims[:3]  # Top 3 weakest areas

    return f"""Mutate the following prompts to improve interview material quality.

## Current Question Prompt
{original_question_prompt}

## Current Rubric Prompt Additions
{current_rubric_additions if current_rubric_additions else "(None yet)"}

## Judge Feedback

### Scores (0-100)
- Question Clarity: {scores.question_clarity}
- Question Relevance: {scores.question_relevance}
- Question Depth: {scores.question_depth}
- Question Coverage: {scores.question_coverage}
- Question Fairness: {scores.question_fairness}
- Rubric Objectivity: {scores.rubric_objectivity}
- Rubric Discrimination: {scores.rubric_discrimination}
- Rubric Calibration: {scores.rubric_calibration}
- Rubric LLM Compatibility: {scores.rubric_llm_compatibility}
- Alignment: {scores.alignment}
- Completeness: {scores.completeness}

### Weakest Areas (prioritize these)
{chr(10).join(f"- {dim}: {score}" for dim, score in weakest)}

### Question Strengths
{chr(10).join(f"- {s}" for s in feedback.question_strengths)}

### Question Weaknesses
{chr(10).join(f"- {w}" for w in feedback.question_weaknesses)}

### Rubric Strengths
{chr(10).join(f"- {s}" for s in feedback.rubric_strengths)}

### Rubric Weaknesses
{chr(10).join(f"- {w}" for w in feedback.rubric_weaknesses)}

### Alignment Issues
{chr(10).join(f"- {i}" for i in feedback.alignment_issues)}

### Improvement Suggestions
{chr(10).join(f"- {s}" for s in feedback.improvement_suggestions)}

## Your Task
1. Create an improved question prompt that addresses weaknesses while keeping strengths
2. Create rubric prompt additions that improve rubric quality
3. Ensure the changes improve alignment between questions and rubric
4. Explain your mutations

Focus especially on the weakest areas identified above.
"""
