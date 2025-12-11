"""Rubric Writer Agent - generates scoring rubrics for question sets."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..models import QuestionSet, ScoringRubric


class RubricOutput(BaseModel):
    """Output from the rubric writer agent."""

    rubric: ScoringRubric = Field(description="The generated scoring rubric")
    calibration_notes: str = Field(
        description="Notes on how to calibrate scoring across interviewers"
    )


RUBRIC_WRITER_SYSTEM_PROMPT = """You are an expert in interview assessment design. Your role is to
create clear, fair, and actionable scoring rubrics that enable consistent candidate evaluation.

## Your Expertise
- You understand how to translate interview performance into objective scores
- You design rubrics that minimize interviewer bias
- You create clear distinctions between score levels
- You balance quantitative scoring with qualitative insights

## Rubric Design Principles
1. **Objectivity**: Criteria should be observable and measurable
2. **Clarity**: Score descriptions should be unambiguous
3. **Discrimination**: Rubric should distinguish between skill levels
4. **Calibration**: Examples should help interviewers score consistently
5. **Actionability**: Feedback should be useful for hiring decisions

## Scoring Approach
- Use a 1-5 scale for each criterion
- 1 = Poor/Unacceptable
- 2 = Below expectations
- 3 = Meets expectations
- 4 = Exceeds expectations
- 5 = Exceptional/Outstanding

## For Each Question's Rubric
- Identify 2-4 key criteria to score
- Provide clear anchors for scores 1, 3, and 5
- Include example excellent and poor answers
- Note red flags that indicate concerns
- Note bonus indicators of exceptional candidates

## LLM Scoring Compatibility
Design the rubric so it can be used by an LLM to score candidate transcript answers:
- Criteria descriptions should be specific enough for automated matching
- Example answers should help calibrate LLM scoring
- Red flags and bonus indicators should be concrete and observable in transcripts

Be thorough but practical. The rubric should work for both human and LLM evaluation.
"""


rubric_writer_agent = Agent(
    "anthropic:claude-sonnet-4-5",
    system_prompt=RUBRIC_WRITER_SYSTEM_PROMPT,
    output_type=RubricOutput,
    model_settings={"temperature": 0.5},
)


def create_rubric_prompt(question_set: QuestionSet, question_prompt: str) -> str:
    """Create a prompt for generating a rubric from a question set."""
    questions_summary = "\n".join(
        f"- Q{i+1} ({q.question_id}): {q.question_text[:100]}..."
        if len(q.question_text) > 100
        else f"- Q{i+1} ({q.question_id}): {q.question_text}"
        for i, q in enumerate(question_set.questions)
    )

    return f"""Create a comprehensive scoring rubric for the following interview question set.

## Original Question Set Design Prompt
{question_prompt}

## Question Set Details
Title: {question_set.title}
Target Role: {question_set.target_role}
Description: {question_set.description}

## Questions
{questions_summary}

## Requirements
1. Create a rubric entry for EACH question
2. Ensure criteria align with the role requirements
3. Provide calibration examples that would help an LLM score candidate answers
4. Include practical red flags and bonus indicators
5. Set appropriate hiring recommendation thresholds

The rubric will be used to score transcripts of candidate answers using an LLM-as-judge.
"""
