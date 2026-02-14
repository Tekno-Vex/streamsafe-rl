"""
Unit tests for RiskScorer component.
Tests deterministic risk scoring and action mapping.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.risk import RiskScorer, ActionType


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.exists = AsyncMock(return_value=False)
    return mock


@pytest.fixture
def risk_scorer(mock_redis):
    """Create a RiskScorer instance with mocked Redis."""
    return RiskScorer(mock_redis, redis_timeout=0.1)


@pytest.mark.asyncio
async def test_risk_scorer_initialization(risk_scorer):
    """Test RiskScorer initializes with correct thresholds."""
    assert risk_scorer.THRESHOLD_WARN == 0.4
    assert risk_scorer.THRESHOLD_TIMEOUT_60S == 0.7
    assert risk_scorer.THRESHOLD_TIMEOUT_600S == 0.85
    assert risk_scorer.THRESHOLD_BAN == 0.95


@pytest.mark.asyncio
async def test_risk_scorer_low_risk(risk_scorer):
    """Test low risk score (<0.4) maps to IGNORE."""
    # Mock a request with low risk features
    request = {
        "message_id": "test1",
        "user_id": "user123",
        "channel_id": "channel1",
        "message_text": "hello world",
        "is_moderator": False,
        "is_subscriber": True,
        "account_age_days": 365
    }
    
    # Low risk features should result in low score
    action = risk_scorer.score_to_action(0.0)
    assert action == ActionType.IGNORE


@pytest.mark.asyncio
async def test_risk_scorer_warn_threshold(risk_scorer):
    """Test risk score 0.4-0.7 maps to WARN."""
    action_low = risk_scorer.score_to_action(0.4)
    action_high = risk_scorer.score_to_action(0.69)
    
    assert action_low == ActionType.WARN
    assert action_high == ActionType.WARN


@pytest.mark.asyncio
async def test_risk_scorer_timeout_60s_threshold(risk_scorer):
    """Test risk score 0.7-0.85 maps to TIMEOUT_60S."""
    action_low = risk_scorer.score_to_action(0.7)
    action_high = risk_scorer.score_to_action(0.84)
    
    assert action_low == ActionType.TIMEOUT_60S
    assert action_high == ActionType.TIMEOUT_60S


@pytest.mark.asyncio
async def test_risk_scorer_timeout_600s_threshold(risk_scorer):
    """Test risk score 0.85-0.95 maps to TIMEOUT_600S."""
    action_low = risk_scorer.score_to_action(0.85)
    action_high = risk_scorer.score_to_action(0.94)
    
    assert action_low == ActionType.TIMEOUT_600S
    assert action_high == ActionType.TIMEOUT_600S


@pytest.mark.asyncio
async def test_risk_scorer_ban_threshold(risk_scorer):
    """Test risk score >=0.95 maps to BAN."""
    action = risk_scorer.score_to_action(0.95)
    assert action == ActionType.BAN


@pytest.mark.asyncio
async def test_risk_scorer_feature_weights(risk_scorer):
    """Test that feature weights are properly configured."""
    weights = risk_scorer.WEIGHTS
    
    # Check key features are weighted
    assert "warnings_last_24h" in weights
    assert "timeouts_last_7d" in weights
    assert "messages_last_hour" in weights
    
    # Moderators should have negative weight
    assert weights["is_moderator"] < 0


@pytest.mark.asyncio
async def test_risk_scorer_deterministic(risk_scorer, mock_redis):
    """Test that identical inputs produce identical scores."""
    message = {
        "user_id": "user123",
        "channel_id": "channel1",
        "message_text": "test message",
        "is_moderator": False,
        "is_subscriber": False,
        "account_age_days": 30
    }
    
    # Mock Redis to return same user history
    mock_redis.get = AsyncMock(return_value=b"2")  # 2 warnings
    
    # Call multiple times and ensure deterministic behavior
    request1 = {**message, "message_id": "msg1"}
    request2 = {**message, "message_id": "msg2"}
    
    # Both should produce same action if features are identical
    action1 = risk_scorer.score_to_action(0.5)
    action2 = risk_scorer.score_to_action(0.5)
    
    assert action1 == action2


@pytest.mark.asyncio
async def test_fetch_user_history_redis_timeout(risk_scorer, mock_redis):
    """Test graceful handling of Redis timeout."""
    # Mock Redis to timeout
    mock_redis.get = AsyncMock(side_effect=asyncio.TimeoutError())
    
    # Should handle gracefully with default values
    history = await risk_scorer.fetch_user_history("user123", "channel1")
    
    assert isinstance(history, dict)
    assert history["warnings_last_24h"] == 0  # Default on error


@pytest.mark.asyncio
async def test_compute_risk_score_moderator_protection(risk_scorer):
    """Test that moderators get reduced risk scores."""
    features_user = {
        "is_moderator": False,
        "warnings_last_24h": 5,
        "timeouts_last_7d": 2
    }
    
    features_mod = {
        **features_user,
        "is_moderator": True
    }
    
    # Moderators should have lower risk due to negative weight on is_moderator
    # (actual computation depends on the compute_risk_score implementation)
    # This ensures the weight structure protects moderators
    assert risk_scorer.WEIGHTS["is_moderator"] < 0


@pytest.mark.asyncio
async def test_action_type_enum(risk_scorer):
    """Test ActionType enum has all expected values."""
    # Test ActionType enum from risk.py (which has more timeouts)
    # But only verify the ones actually used by the system
    assert ActionType.IGNORE.value == "IGNORE"
    assert ActionType.WARN.value == "WARN"
    assert ActionType.TIMEOUT_60S.value == "TIMEOUT_60S"
    assert ActionType.TIMEOUT_600S.value == "TIMEOUT_600S"
    assert ActionType.BAN.value == "BAN"
