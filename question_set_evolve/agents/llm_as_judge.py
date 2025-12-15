"""LLM-as-Judge Agent - evaluates quality of question sets and rubrics."""

from __future__ import annotations

from pydantic_ai import Agent

from ..models import QuestionSet, ScoringRubric, QualityScores, JudgeFeedback


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of interview materials. Your role is to
critically assess interview question sets and scoring rubrics for quality and effectiveness.

## Evaluation Framework

### Question Set Quality Dimensions
1. **Clarity (0-100)**: Are questions unambiguous? Will candidates understand what's being asked?
2. **Relevance (0-100)**: Do questions assess skills actually needed for the role?
3. **Depth (0-100)**: Can strong candidates demonstrate expertise? Are there layers to explore?
4. **Coverage (0-100)**: Are all important skill areas represented?
5. **Fairness (0-100)**: Are questions free from bias? Accessible to diverse candidates?

### Rubric Quality Dimensions
1. **Objectivity (0-100)**: Are criteria observable and measurable?
2. **Discrimination (0-100)**: Does the rubric distinguish between skill levels?
3. **Calibration (0-100)**: Do examples help scorers be consistent?
4. **LLM Compatibility (0-100)**: Can an LLM reliably use this rubric to score transcripts?

### Co-evolution Quality
1. **Alignment (0-100)**: Do rubric criteria map clearly to question requirements?
2. **Completeness (0-100)**: Does every question have adequate rubric coverage?

## Evaluation Approach
- Be critical but constructive
- Provide specific examples when noting issues
- Consider practical interview settings
- Think about both human and LLM scorers
- Balance thoroughness with usability

Score honestly. A score of 70-80 represents solid, usable materials. Reserve 90+ for exceptional work.
"""


judge_agent = Agent[None, JudgeFeedback](
    "anthropic:claude-haiku-4-5",
    output_type=JudgeFeedback,
    instructions=JUDGE_SYSTEM_PROMPT,
    model_settings={"temperature": 0.1, "max_tokens": 8192},
    retries=3,
)


def create_judge_prompt(
    question_set: QuestionSet,
    rubric: ScoringRubric,
    original_prompt: str,
) -> str:
    """Create a prompt for judging a question set and rubric pair."""
    return f"""Evaluate the following interview question set and scoring rubric.

## Original Design Prompt
{original_prompt}

## Question Set
Title: {question_set.title}
Target Role: {question_set.target_role}
Description: {question_set.description}
Total Time: {question_set.total_time_minutes} minutes

### Questions
{_format_questions(question_set)}

## Scoring Rubric
{_format_rubric(rubric)}

## Your Task
1. Score each quality dimension (0-100)
2. Summarize what works well (strengths)
3. Describe key issues and how to improve them (improvements)

Focus on whether these materials would work well in practice for human interviewers and LLM judges.
"""


def _format_questions(question_set: QuestionSet) -> str:
    """Format questions for display."""
    parts = []
    for i, q in enumerate(question_set.questions, 1):
        parts.append(
            f"""
**Q{i} ({q.question_id})** [{q.category}] [{q.difficulty}] [{q.time_allocation_minutes} min]
{q.question_text}

Follow-ups: {', '.join(q.follow_up_questions) if q.follow_up_questions else 'None'}
Look for: {q.what_to_look_for}
"""
        )
    return "\n".join(parts)


def _format_rubric(rubric: ScoringRubric) -> str:
    """Format rubric for display."""
    parts = [f"**Overall Guidance**: {rubric.overall_scoring_guidance}\n"]

    for qr in rubric.question_rubrics:
        parts.append(f"\n### Rubric for {qr.question_id}")
        for c in qr.criteria:
            parts.append(
                f"""
**{c.name}** (weight: {c.weight})
- {c.description}
- Score 1: {c.score_1_description}
- Score 3: {c.score_3_description}
- Score 5: {c.score_5_description}
"""
            )
        parts.append(f"Excellent answer example: {qr.example_excellent_answer[:200]}...")
        parts.append(f"Poor answer example: {qr.example_poor_answer[:200]}...")
        if qr.red_flags:
            parts.append(f"Red flags: {', '.join(qr.red_flags)}")
        if qr.bonus_indicators:
            parts.append(f"Bonus indicators: {', '.join(qr.bonus_indicators)}")

    thresholds = rubric.hiring_recommendation_thresholds
    parts.append(
        f"\n**Hiring Thresholds**: Strong Hire >= {thresholds.get('strong_hire', 4.5)}, "
        f"Hire >= {thresholds.get('hire', 3.5)}, "
        f"Lean No Hire >= {thresholds.get('lean_no_hire', 2.5)}"
    )

    return "\n".join(parts)
