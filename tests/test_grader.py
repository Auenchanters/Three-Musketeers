"""
Tests for engine/grader.py — Oracle grading formulas.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.grader import Grader


class TestSafetyMultiplier:
    def test_no_violations(self):
        assert Grader.compute_safety_multiplier([], "easy") == 1.0
        assert Grader.compute_safety_multiplier([], "medium") == 1.0
        assert Grader.compute_safety_multiplier([], "hard") == 1.0

    def test_production_violation(self):
        violations = ["Deleted production/critical resource i-abc123"]
        assert Grader.compute_safety_multiplier(violations, "easy") == 0.0
        assert Grader.compute_safety_multiplier(violations, "medium") == 0.0
        assert Grader.compute_safety_multiplier(violations, "hard") == 0.0

    def test_critical_violation(self):
        violations = ["Deleted critical resource i-abc123"]
        assert Grader.compute_safety_multiplier(violations, "easy") == 0.0

    def test_staging_violation_medium(self):
        violations = ["Deleted staging resource vol-abc123"]
        assert Grader.compute_safety_multiplier(violations, "medium") == 0.7
        assert Grader.compute_safety_multiplier(violations, "hard") == 0.7


class TestFinalScore:
    def test_perfect_easy(self):
        """Perfect easy score: save everything, no violations."""
        score = Grader.compute_final_score(
            actual_savings=100.0,
            optimal_savings=100.0,
            steps_taken=10,
            safety_violations=[],
            difficulty="easy",
        )
        assert score == 1.0

    def test_zero_savings(self):
        score = Grader.compute_final_score(
            actual_savings=0.0,
            optimal_savings=100.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="easy",
        )
        assert score == 0.0

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
        assert score == 0.0

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
        expected = 1.0 - (10 * 0.003)  # 0.97
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

    def test_score_clamped_to_0_1(self):
        """Scores should always be in [0, 1]."""
        # Very negative case
        score = Grader.compute_final_score(
            actual_savings=0.0,
            optimal_savings=100.0,
            steps_taken=200,
            safety_violations=[],
            difficulty="medium",
        )
        assert score >= 0.0

        # Very positive case (shouldn't exceed 1)
        score = Grader.compute_final_score(
            actual_savings=200.0,
            optimal_savings=100.0,
            steps_taken=1,
            safety_violations=[],
            difficulty="easy",
        )
        assert score <= 1.0

    def test_zero_optimal_savings(self):
        """Edge case: optimal_savings = 0 should return 0."""
        score = Grader.compute_final_score(
            actual_savings=50.0,
            optimal_savings=0.0,
            steps_taken=5,
            safety_violations=[],
            difficulty="easy",
        )
        assert score == 0.0
