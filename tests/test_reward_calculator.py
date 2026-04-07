import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.reward_calculator import RewardCalculator

def test_query_metrics_reward():
    reward = RewardCalculator.query_metrics_reward()
    assert reward.value < 0
    assert "investigation_cost" in reward.breakdown
    assert "Queried" in reward.message

def test_successful_delete_reward():
    reward = RewardCalculator.successful_delete_reward("vol-1", 50.0)
    assert reward.value > 0
    assert reward.breakdown["savings"] > 0
    assert "Saving $50.00" in reward.message

def test_delete_production_reward():
    reward = RewardCalculator.delete_production_reward("i-prod")
    assert reward.value < -0.9
    assert reward.breakdown["safety_penalty"] <= -1.0
    assert "CRITICAL" in reward.message

def test_commit_reward_excellent():
    reward = RewardCalculator.commit_reward(90.0, 100.0)
    assert reward.value > 0.2
    assert "Excellent" in reward.message

def test_commit_reward_good():
    reward = RewardCalculator.commit_reward(60.0, 100.0)
    assert reward.value > 0.1
    assert "Good" in reward.message

def test_commit_reward_zero_optimal():
    reward = RewardCalculator.commit_reward(0.0, 0.0)
    assert reward.value == 0.0
    assert reward.breakdown["savings_ratio"] == 0.0
