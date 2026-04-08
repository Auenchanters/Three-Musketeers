"""
Tests for engine/grader.py — Oracle grading formulas.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.grader import Grader

# Mirrors the epsilon used by the grader to clamp scores into the open
# interval (0, 1) — keep in sync with engine/grader.py.
SCORE_EPS = 0.01


class TestSafetyMultiplier:
    def test_no_violations(self):
        # The grader rolls the score-epsilon into the safety multiplier so
        # that perfect runs land at 1 - SCORE_EPS instead of exactly 1.0.
        assert Grader.compute_safety_multiplier([], "easy") == pytest.approx(1.0 - SCORE_EPS)
        assert Grader.compute_safety_multiplier([], "medium") == pytest.approx(1.0 - SCORE_EPS)
        assert Grader.compute_safety_multiplier([], "hard") == pytest.approx(1.0 - SCORE_EPS)

    def test_production_violation(self):
        violations = ["Deleted production/critical resource i-abc123"]
        assert Grader.compute_safety_multiplier(violations, "easy") == pytest.approx(SCORE_EPS)
        assert Grader.compute_safety_multiplier(violations, "medium") == pytest.approx(SCORE_EPS)
        assert Grader.compute_safety_multiplier(violations, "hard") == pytest.approx(SCORE_EPS)

    def test_critical_violation(self):
        violations = ["Deleted critical resource i-abc123"]
        assert Grader.compute_safety_multiplier(violations, "easy") == pytest.approx(SCORE_EPS)

    def test_staging_violation_medium(self):
        violations = ["Deleted staging resource vol-abc123"]
        assert Grader.compute_safety_multiplier(violations, "medium") == 0.7
        assert Grader.compute_safety_multiplier(violations, "hard") == 0.7


class TestFinalScore:
    def test_perfect_easy(self):
        """Perfect easy score: save everything, no violations.

        Scores must be strictly inside (0, 1), so a perfect run clamps to
        the upper epsilon bound (1 - SCORE_EPS) instead of exactly 1.0.
        """
        score = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=10,
            safety_violations=[],
            difficulty="easy",
        )
        assert score == pytest.approx(1.0 - SCORE_EPS)
        assert 0.0 < score < 1.0

    def test_zero_savings(self):
        score = Grader.compute_final_score(
            actual_savings=0.0,
            optimal_savings=100.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="easy",
        )
        assert score == pytest.approx(SCORE_EPS)
        assert 0.0 < score < 1.0

    def test_partial_savings_easy(self):
        score = Grader.compute_final_score(
            actual_savings=60.0,
            optimal_savings=100.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="easy",
        )
        assert 0.55 <= score <= 0.65  # 60% savings

    def test_production_violation_zeroes_score(self):
        score = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=5,
            safety_violations=["Deleted production resource i-abc"],
            difficulty="easy",
        )
        assert score == pytest.approx(SCORE_EPS)
        assert 0.0 < score < 1.0

    def test_medium_step_penalty(self):
        """Medium has 0.005 per step penalty."""
        score_few = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="medium",
        )
        score_many = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=20,
            safety_violations=[],
            difficulty="medium",
        )
        assert score_few > score_many
        assert abs(score_few - score_many - 15 * 0.005) < 0.001

    def test_hard_step_penalty(self):
        """Hard has 0.003 per step penalty."""
        score = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=10,
            safety_violations=[],
            difficulty="hard",
        )
        # Perfect-run baseline is (1 - SCORE_EPS) due to the epsilon-rolled
        # safety multiplier, so a 10-step hard run lands at 0.96, not 0.97.
        expected = (1.0 - SCORE_EPS) - (10 * 0.003)  # 0.96
        assert abs(score - expected) < 0.001

    def test_hard_cascade_penalty(self):
        """Hard subtracts cascade penalty from savings."""
        score_no_cascade = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="hard",
            cascade_penalty=0.0,
        )
        score_with_cascade = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="hard",
            cascade_penalty=20.0,
        )
        assert score_no_cascade > score_with_cascade

    def test_score_clamped_to_open_unit_interval(self):
        """Scores must always be strictly in (0, 1) — never exactly 0 or 1."""
        # Very negative case
        score = Grader.compute_final_score(
            actual_savings=0.0,
            optimal_savings=100.0,
            steps_taken=200,
            safety_violations=[],
            difficulty="medium",
        )
        assert 0.0 < score < 1.0

        # Very positive case (shouldn't reach 1)
        score = Grader.compute_final_score(
            actual_savings=200.0,
            optimal_savings=100.0,
            steps_taken=1,
            safety_violations=[],
            difficulty="easy",
        )
        assert 0.0 < score < 1.0

    def test_zero_optimal_savings(self):
        """Edge case: optimal_savings = 0 collapses to the lower epsilon bound."""
        score = Grader.compute_final_score(
            actual_savings=50.0,
            optimal_savings=0.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="easy",
        )
        assert score == pytest.approx(SCORE_EPS)
        assert 0.0 < score < 1.0
