"""Data models for question set evolution."""

from pydantic import BaseModel, Field
from typing import Optional


class InterviewQuestion(BaseModel):
    """A single interview question with metadata."""

    question_id: str = Field(default="", description="Unique identifier for this question")
    question_text: str = Field(default="", description="The actual question to ask the candidate")
    category: str = Field(default="general", description="Category/topic area (e.g., 'system design', 'behavioral')")
    difficulty: str = Field(default="medium", description="Difficulty level: easy, medium, hard")
    time_allocation_minutes: int = Field(default=10, description="Suggested time in minutes for this question")
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Optional follow-up questions to probe deeper",
    )
    what_to_look_for: str = Field(
        default="",
        description="Brief guidance for interviewers on what good answers include"
    )


class QuestionSet(BaseModel):
    """A complete set of interview questions."""

    title: str = Field(default="Interview Question Set", description="Title of the question set")
    description: str = Field(default="", description="Description of what this question set evaluates")
    target_role: str = Field(default="", description="The role this question set is designed for")
    total_time_minutes: int = Field(default=60, description="Total interview time in minutes")
    questions: list[InterviewQuestion] = Field(default_factory=list, description="The list of questions")


class ScoringCriterion(BaseModel):
    """A single criterion for scoring a candidate's answer."""

    criterion_id: str = Field(default="", description="Unique identifier for this criterion")
    name: str = Field(default="", description="Short name for this criterion")
    description: str = Field(default="", description="What this criterion measures")
    weight: float = Field(default=1.0, description="Relative weight of this criterion (0.0-2.0)")
    score_1_description: str = Field(default="Poor performance", description="What a score of 1 (poor) looks like")
    score_3_description: str = Field(default="Acceptable performance", description="What a score of 3 (acceptable) looks like")
    score_5_description: str = Field(default="Excellent performance", description="What a score of 5 (excellent) looks like")


class QuestionRubric(BaseModel):
    """Scoring rubric for a specific question."""

    question_id: str = Field(default="", description="ID of the question this rubric scores")
    criteria: list[ScoringCriterion] = Field(default_factory=list, description="Scoring criteria for this question")
    example_excellent_answer: str = Field(
        default="", description="Example of an excellent answer (for calibration)"
    )
    example_poor_answer: str = Field(
        default="", description="Example of a poor answer (for calibration)"
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Answers or behaviors that are immediate concerns",
    )
    bonus_indicators: list[str] = Field(
        default_factory=list,
        description="Indicators that suggest exceptional candidates",
    )


class ScoringRubric(BaseModel):
    """Complete scoring rubric for a question set."""

    title: str = Field(default="Scoring Rubric", description="Title matching the question set")
    overall_scoring_guidance: str = Field(
        default="", description="General guidance for scoring candidates"
    )
    question_rubrics: list[QuestionRubric] = Field(
        default_factory=list, description="Individual rubrics for each question"
    )
    hiring_recommendation_thresholds: dict[str, float] = Field(
        description="Score thresholds for hire/no-hire decisions",
        default_factory=lambda: {
            "strong_hire": 4.5,
            "hire": 3.5,
            "lean_no_hire": 2.5,
            "no_hire": 0.0,
        },
    )


class CandidateScore(BaseModel):
    """Score for a single criterion."""

    criterion_id: str = Field(default="")
    score: int = Field(default=3, ge=1, le=5, description="Score from 1-5")
    justification: str = Field(default="", description="Brief justification for this score")


class QuestionScore(BaseModel):
    """Scores for all criteria on a single question."""

    question_id: str = Field(default="")
    criterion_scores: list[CandidateScore] = Field(default_factory=list)
    overall_question_score: float = Field(default=3.0, description="Weighted average for this question")
    notes: str = Field(default="", description="Additional notes about this answer")


class CandidateEvaluation(BaseModel):
    """Complete evaluation of a candidate's interview."""

    candidate_id: str = Field(default="", description="Identifier for the candidate")
    question_scores: list[QuestionScore] = Field(default_factory=list)
    overall_score: float = Field(default=3.0, description="Overall weighted score")
    strengths: list[str] = Field(default_factory=list, description="Key strengths demonstrated")
    areas_for_improvement: list[str] = Field(default_factory=list, description="Areas needing development")
    hiring_recommendation: str = Field(
        default="lean_no_hire", description="One of: strong_hire, hire, lean_no_hire, no_hire"
    )
    summary: str = Field(default="", description="Brief narrative summary of the candidate")
