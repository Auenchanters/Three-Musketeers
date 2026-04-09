"""Deterministic scoring formulas for each difficulty level."""

from typing import List


class Grader:
    """
    Oracle grader with difficulty-specific scoring formulas.
    
    Easy:   score = (actual / optimal) * safety_mult
    Medium: score = (actual / optimal) * safety_mult - (steps * 0.005)
    Hard:   score = ((actual - cascade_pen) / optimal) * safety_mult - (steps * 0.003)
    
    All scores clamped to (0.01, 0.99) — strictly within (0, 1) as required by validator.
    """

    @staticmethod
    def compute_safety_multiplier(
        safety_violations: List[str],
        difficulty: str,
    ) -> float:
        """
        Compute the safety multiplier based on violations.
        
        - Any production resource deleted → 0.0
        - Any staging resource wrongly deleted (medium/hard) → 0.7
        - No violations → 1.0
        """
        if not safety_violations:
            return 0.99

        # Check for production violations (catastrophic)
        for v in safety_violations:
            if "production" in v.lower() or "critical" in v.lower():
                return 0.01

        # Non-production violations (staging mistakes etc.)
        if difficulty in ("medium", "hard"):
            return 0.7  # already within (0, 1)

        return 0.99

    @staticmethod
    def compute_final_score(
        actual_savings: float,
        optimal_savings: float,
        steps_taken: int,
        safety_violations: List[str],
        difficulty: str,
        cascade_penalty: float = 0.0,
    ) -> float:
        """
        Compute the final episode score.
        
        Args:
            actual_savings: Total monthly savings achieved.
            optimal_savings: Maximum possible monthly savings (oracle).
            steps_taken: Number of steps the agent took.
            safety_violations: List of safety violation messages.
            difficulty: "easy", "medium", or "hard".
            cascade_penalty: Cost of unintended cascading side-effects (hard only).
            
        Returns:
            Score clamped to (0.01, 0.99) — strictly within (0, 1).
        """
        if optimal_savings <= 0:
            return 0.01

        safety_mult = Grader.compute_safety_multiplier(safety_violations, difficulty)

        if difficulty == "easy":
            raw_score = (actual_savings / optimal_savings) * safety_mult

        elif difficulty == "medium":
            raw_score = (
                (actual_savings / optimal_savings) * safety_mult
                - (steps_taken * 0.005)
            )

        elif difficulty == "hard":
            raw_score = (
                ((actual_savings - cascade_penalty) / optimal_savings) * safety_mult
                - (steps_taken * 0.003)
            )

        else:
            raw_score = (actual_savings / optimal_savings) * safety_mult

        # Final clamping to strictly inside (0, 1) for validator compliance.
        # Handles NaN/Inf by defaulting to 0.01.
        try:
            clamped_score = float(raw_score)
            if clamped_score != clamped_score:  # NaN check
                clamped_score = 0.01
            return round(min(max(clamped_score, 0.01), 0.99), 4)
        except (ValueError, TypeError):
            return 0.01
