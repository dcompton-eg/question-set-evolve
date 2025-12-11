"""Tests for data models."""

import pytest
from question_set_evolve.models import (
    InterviewQuestion,
    QuestionSet,
    ScoringCriterion,
    QuestionRubric,
    ScoringRubric,
)


def test_interview_question_creation():
    """Test creating an interview question."""
    q = InterviewQuestion(
        question_id="q1",
        question_text="Tell me about a challenging project.",
        category="behavioral",
        difficulty="medium",
        time_allocation_minutes=10,
        what_to_look_for="Problem-solving approach and communication skills",
    )
    assert q.question_id == "q1"
    assert q.difficulty == "medium"
    assert q.follow_up_questions == []


def test_question_set_creation():
    """Test creating a question set."""
    q1 = InterviewQuestion(
        question_id="q1",
        question_text="Design a URL shortener.",
        category="system_design",
        difficulty="hard",
        time_allocation_minutes=20,
        what_to_look_for="Scalability considerations",
    )
    qs = QuestionSet(
        title="Backend Engineer Interview",
        description="Technical interview for backend roles",
        target_role="Senior Backend Engineer",
        total_time_minutes=60,
        questions=[q1],
    )
    assert len(qs.questions) == 1
    assert qs.total_time_minutes == 60


def test_scoring_criterion():
    """Test scoring criterion validation."""
    c = ScoringCriterion(
        criterion_id="c1",
        name="Technical Depth",
        description="Demonstrates deep understanding",
        weight=1.5,
        score_1_description="Surface-level understanding only",
        score_3_description="Solid understanding with some gaps",
        score_5_description="Expert-level knowledge demonstrated",
    )
    assert c.weight == 1.5
    assert c.criterion_id == "c1"


def test_question_rubric():
    """Test question rubric creation."""
    criterion = ScoringCriterion(
        criterion_id="c1",
        name="Problem Solving",
        description="Ability to break down and solve problems",
        score_1_description="Cannot break down problem",
        score_3_description="Basic problem decomposition",
        score_5_description="Elegant problem decomposition",
    )
    rubric = QuestionRubric(
        question_id="q1",
        criteria=[criterion],
        example_excellent_answer="The candidate methodically analyzed...",
        example_poor_answer="The candidate jumped to solutions without...",
        red_flags=["Dismisses edge cases"],
        bonus_indicators=["Considers multiple approaches"],
    )
    assert rubric.question_id == "q1"
    assert len(rubric.criteria) == 1
    assert len(rubric.red_flags) == 1


def test_scoring_rubric_thresholds():
    """Test scoring rubric with default thresholds."""
    rubric = ScoringRubric(
        title="Test Rubric",
        overall_scoring_guidance="Score based on demonstrated skills",
        question_rubrics=[],
    )
    assert rubric.hiring_recommendation_thresholds["strong_hire"] == 4.5
    assert rubric.hiring_recommendation_thresholds["hire"] == 3.5
