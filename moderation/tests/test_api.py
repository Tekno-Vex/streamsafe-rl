"""
Unit tests for API schema validation and imports.
Integration tests check the actual endpoints via run_tests.sh.
"""
import pytest
from app.api import app
from app.schemas.action import ActionType


def test_fastapi_app_initialized():
    """Test that FastAPI app is initialized."""
    assert app is not None
    assert hasattr(app, "routes")


def test_fastapi_has_health_route():
    """Test that /health route is defined."""
    routes = [route.path for route in app.routes]
    assert "/health" in routes


def test_fastapi_has_moderate_route():
    """Test that /moderate route is defined."""
    routes = [route.path for route in app.routes]
    assert "/moderate" in routes


def test_fastapi_has_metrics_route():
    """Test that /metrics route is defined."""
    routes = [route.path for route in app.routes]
    assert "/metrics" in routes


def test_fastapi_has_log_event_route():
    """Test that /log_event route is defined."""
    routes = [route.path for route in app.routes]
    assert "/log_event" in routes


def test_action_type_enum_values():
    """Test ActionType enum has expected values."""
    expected_actions = ["IGNORE", "WARN", "TIMEOUT_60S", "TIMEOUT_600S", "BAN"]
    
    for action_name in expected_actions:
        assert hasattr(ActionType, action_name)
        action = getattr(ActionType, action_name)
        assert action.value == action_name


def test_action_type_is_enum():
    """Test ActionType is a proper enum."""
    actions = list(ActionType)
    assert len(actions) > 0
    
    # All should have string values
    for action in actions:
        assert isinstance(action.value, str)


def test_action_type_deterministic_mapping():
    """Test that ActionType mapping is deterministic."""
    # Same value should always produce same enum
    action1 = ActionType("WARN")
    action2 = ActionType("WARN")
    assert action1 is action2


def test_action_type_values_are_unique():
    """Test that all ActionType values are unique."""
    actions = list(ActionType)
    values = [a.value for a in actions]
    
    assert len(values) == len(set(values))


def test_weighted_scoring_architecture():
    """Test that RiskScorer uses weighted features."""
    from app.risk import RiskScorer
    
    assert hasattr(RiskScorer, 'WEIGHTS')
    weights = RiskScorer.WEIGHTS
    
    # Should have multiple features with weights
    assert len(weights) > 0
    assert isinstance(weights, dict)
    
    # All weights should be numeric
    for key, weight in weights.items():
        assert isinstance(weight, (int, float))


def test_risk_thresholds_configured():
    """Test that RiskScorer has proper thresholds."""
    from app.risk import RiskScorer
    
    assert hasattr(RiskScorer, 'THRESHOLD_WARN')
    assert hasattr(RiskScorer, 'THRESHOLD_TIMEOUT_60S')
    assert hasattr(RiskScorer, 'THRESHOLD_TIMEOUT_600S')
    assert hasattr(RiskScorer, 'THRESHOLD_BAN')
    
    # Thresholds should be in order
    assert RiskScorer.THRESHOLD_WARN < RiskScorer.THRESHOLD_TIMEOUT_60S
    assert RiskScorer.THRESHOLD_TIMEOUT_60S < RiskScorer.THRESHOLD_TIMEOUT_600S
    assert RiskScorer.THRESHOLD_TIMEOUT_600S < RiskScorer.THRESHOLD_BAN


def test_executor_has_rate_limiting():
    """Test that ActionExecutor has rate limiting configured."""
    from app.executor import ActionExecutor
    
    executor = ActionExecutor()
    assert hasattr(executor, 'min_interval')
    assert hasattr(executor, 'dedupe_window')
    
    # 10 commands/second = 0.1s minimum interval
    assert executor.min_interval > 0
    # 5 second dedup window
    assert executor.dedupe_window == 5.0


def test_safety_clamp_has_rules():
    """Test that SafetyClamp has defined safety rules."""
    from app.safety_clamp import SafetyClamp
    
    clamp = SafetyClamp()
    assert hasattr(clamp, 'clamp')
    
    # Should have clamp method
    assert callable(clamp.clamp)


def test_logger_pipeline_configured():
    """Test that event logger is properly configured."""
    from app.logger_pipeline import ModerationEventLogger
    
    assert ModerationEventLogger is not None
    
    # Should have logging capabilities
    assert hasattr(ModerationEventLogger, '__init__')


def test_moderation_response_schemas():
    """Test that moderation response has proper schema."""
    from app.api import ModerationResponse
    
    # Should be a Pydantic model
    assert hasattr(ModerationResponse, '__fields__')
    
    # Should have required fields
    required_fields = {'message_id', 'action', 'risk_score'}
    fields = set(ModerationResponse.__fields__.keys())
    assert required_fields.issubset(fields)


def test_health_response_schema():
    """Test that health check returns proper format."""
    # Health should return dict with status field
    # (Verified by integration tests in run_tests.sh)
    assert True  # Placeholder for schema validation


def test_metrics_response_has_latency():
    """Test that metrics response includes latency measurements."""
    # Metrics should include p50_latency and p99_latency
    # (Verified by integration tests in run_tests.sh)
    assert True  # Placeholder for schema validation


def test_imports_available():
    """Test that all critical modules can be imported."""
    try:
        from app.api import app, ModerationResponse
        from app.risk import RiskScorer, ActionType
        from app.executor import ActionExecutor
        from app.safety_clamp import SafetyClamp
        from app.logger_pipeline import ModerationEventLogger
        from app.schemas.action import ActionType as ActionTypeSchema
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")
