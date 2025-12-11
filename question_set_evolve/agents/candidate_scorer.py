"""Candidate Scorer Agent - scores candidate transcripts using rubrics."""

from __future__ import annotations

from pydantic_ai import Agent

from ..models import QuestionSet, ScoringRubric, CandidateEvaluation


CANDIDATE_SCORER_SYSTEM_PROMPT = """You are an expert interview evaluator. Your role is to
objectively score candidate interview transcripts using the provided scoring rubric.

## Your Approach

1. **Read Carefully**: Review the entire transcript before scoring
2. **Apply Rubric**: Use the rubric criteria exactly as specified
3. **Be Objective**: Score based on evidence in the transcript only
4. **Be Consistent**: Apply the same standards throughout

## Scoring Process

For each question in the rubric:
1. Find the candidate's response to that question in the transcript
2. Score each criterion from 1-5 based on the rubric anchors
3. Note specific evidence from the transcript supporting your scores
4. Calculate weighted average for the question

## Score Meanings
- 1 = Poor/Unacceptable - Does not meet minimum requirements
- 2 = Below Expectations - Shows some understanding but significant gaps
- 3 = Meets Expectations - Acceptable, competent response
- 4 = Exceeds Expectations - Strong response with good depth
- 5 = Exceptional - Outstanding response demonstrating mastery

## Red Flags and Bonuses
- Note any red flags from the rubric observed in answers
- Note any bonus indicators that suggest exceptional capability

## Fairness Requirements
- Score only on demonstrated knowledge and skills
- Do not penalize communication style differences
- Consider context and clarifying questions asked

Be thorough but fair. Your evaluation should be actionable for hiring decisions.
"""


candidate_scorer_agent = Agent[None, CandidateEvaluation](
    "anthropic:claude-haiku-4-5",
    output_type=CandidateEvaluation,
    instructions=CANDIDATE_SCORER_SYSTEM_PROMPT,
    model_settings={"temperature": 0.1, "max_tokens": 8192},
    retries=3,
)


def create_scoring_prompt(
    question_set: QuestionSet,
    rubric: ScoringRubric,
    transcript: str,
    candidate_id: str = "candidate",
) -> str:
    """Create a prompt for scoring a candidate transcript."""
    return f"""Score the following candidate interview transcript.

## Interview Context
- Role: {question_set.target_role}
- Interview: {question_set.title}
- Description: {question_set.description}

## Questions Asked
{_format_questions(question_set)}

## Scoring Rubric
{_format_rubric(rubric)}

## Candidate Transcript
{transcript}

## Your Task
1. For each question, find the candidate's response in the transcript
2. Score each criterion (1-5) with justification
3. Calculate overall question scores (weighted average)
4. Calculate overall score across all questions
5. Identify strengths and areas for improvement
6. Make a hiring recommendation based on thresholds:
   - Strong Hire: >= {rubric.hiring_recommendation_thresholds.get('strong_hire', 4.5)}
   - Hire: >= {rubric.hiring_recommendation_thresholds.get('hire', 3.5)}
   - Lean No Hire: >= {rubric.hiring_recommendation_thresholds.get('lean_no_hire', 2.5)}
   - No Hire: < {rubric.hiring_recommendation_thresholds.get('lean_no_hire', 2.5)}
7. Write a brief narrative summary

Use candidate_id: "{candidate_id}"
"""


def _format_questions(question_set: QuestionSet) -> str:
    """Format questions for the prompt."""
    parts = []
    for i, q in enumerate(question_set.questions, 1):
        parts.append(f"{i}. [{q.question_id}] {q.question_text}")
        if q.follow_up_questions:
            for fu in q.follow_up_questions:
                parts.append(f"   Follow-up: {fu}")
    return "\n".join(parts)


def _format_rubric(rubric: ScoringRubric) -> str:
    """Format rubric for the prompt."""
    parts = [f"Overall Guidance: {rubric.overall_scoring_guidance}\n"]

    for qr in rubric.question_rubrics:
        parts.append(f"\n### {qr.question_id}")
        for c in qr.criteria:
            parts.append(f"\n**{c.criterion_id}: {c.name}** (weight: {c.weight})")
            parts.append(f"  Description: {c.description}")
            parts.append(f"  Score 1 (Poor): {c.score_1_description}")
            parts.append(f"  Score 3 (Meets): {c.score_3_description}")
            parts.append(f"  Score 5 (Excellent): {c.score_5_description}")

        parts.append(f"\nExcellent example: {qr.example_excellent_answer}")
        parts.append(f"Poor example: {qr.example_poor_answer}")

        if qr.red_flags:
            parts.append(f"Red flags: {', '.join(qr.red_flags)}")
        if qr.bonus_indicators:
            parts.append(f"Bonus indicators: {', '.join(qr.bonus_indicators)}")

    return "\n".join(parts)
