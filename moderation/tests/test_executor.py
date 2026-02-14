"""
Unit tests for ActionExecutor component.
Tests rate limiting and deduplication logic.
"""
import pytest
import asyncio
import time
from app.executor import ActionExecutor
from app.schemas.action import ActionType


@pytest.fixture
def executor():
    """Create an ActionExecutor instance."""
    return ActionExecutor()


@pytest.mark.asyncio
async def test_executor_initialization(executor):
    """Test ActionExecutor initializes with correct rate limiting config."""
    assert executor.min_interval == 0.1  # 10 commands/sec
    assert executor.dedupe_window == 5.0  # 5 second dedup window
    assert executor.last_global_execution == 0.0
    assert executor.user_state == {}


@pytest.mark.asyncio
async def test_executor_ignore_action(executor):
    """Test that IGNORE actions are skipped without logging."""
    initial_state = executor.user_state.copy()
    
    # Execute IGNORE action
    await executor.execute_action("user123", ActionType.IGNORE, "test reason")
    
    # State should not change since IGNORE is skipped
    assert executor.user_state == initial_state


@pytest.mark.asyncio
async def test_executor_deduplication(executor):
    """Test that duplicate actions within 5s are not executed."""
    user_id = "user123"
    action = ActionType.WARN
    
    # First action should execute
    await executor.execute_action(user_id, action, "first")
    assert user_id in executor.user_state
    first_time = executor.user_state[user_id][1]
    
    # Second identical action within 5s should be skipped
    await executor.execute_action(user_id, action, "duplicate")
    second_time = executor.user_state[user_id][1]
    
    # Timestamps should be identical (action was skipped)
    assert first_time == second_time


@pytest.mark.asyncio
async def test_executor_different_action_allowed(executor):
    """Test that different actions to same user within 5s are allowed."""
    user_id = "user123"
    
    # First action
    await executor.execute_action(user_id, ActionType.WARN, "warn")
    first_action = executor.user_state[user_id][0]
    
    # Different action should be allowed
    await executor.execute_action(user_id, ActionType.TIMEOUT_60S, "timeout")
    second_action = executor.user_state[user_id][0]
    
    assert first_action != second_action
    assert second_action == ActionType.TIMEOUT_60S


@pytest.mark.asyncio
async def test_executor_dedup_window_expiry(executor):
    """Test that actions are allowed after dedup window expires."""
    user_id = "user123"
    action = ActionType.WARN
    
    # First action
    await executor.execute_action(user_id, action, "first")
    first_time = executor.user_state[user_id][1]
    
    # Simulate time passing beyond dedup window
    executor.user_state[user_id] = (action, time.time() - 6.0)
    
    # Second action should be allowed (dedup window expired)
    await executor.execute_action(user_id, action, "allowed")
    second_time = executor.user_state[user_id][1]
    
    # Timestamps should be different (action was executed)
    assert second_time > first_time


@pytest.mark.asyncio
async def test_executor_rate_limiting_enforced(executor):
    """Test that global rate limiting is enforced."""
    # Record start time
    start = time.time()
    
    # Execute multiple actions in sequence
    for i in range(3):
        await executor.execute_action(f"user{i}", ActionType.WARN, f"action {i}")
    
    # With min_interval=0.1s, 3 actions should take ~0.2s
    elapsed = time.time() - start
    assert elapsed >= 0.15  # Allow some variation


@pytest.mark.asyncio
async def test_executor_last_global_execution_updated(executor):
    """Test that last_global_execution timestamp is updated."""
    initial_time = executor.last_global_execution
    
    await executor.execute_action("user1", ActionType.WARN, "test")
    
    # Time should be updated
    assert executor.last_global_execution > initial_time


@pytest.mark.asyncio
async def test_executor_multiple_users_independent(executor):
    """Test that deduplication is per-user, not global."""
    action = ActionType.TIMEOUT_60S
    
    # Same action on two different users should both execute
    await executor.execute_action("user1", action, "user1 action")
    await executor.execute_action("user2", action, "user2 action")
    
    assert "user1" in executor.user_state
    assert "user2" in executor.user_state
    
    # Both should have been updated to the action
    assert executor.user_state["user1"][0] == action
    assert executor.user_state["user2"][0] == action


@pytest.mark.asyncio
async def test_executor_ban_action(executor):
    """Test that BAN actions are executed."""
    user_id = "user_to_ban"
    
    await executor.execute_action(user_id, ActionType.BAN, "spamming")
    
    assert user_id in executor.user_state
    assert executor.user_state[user_id][0] == ActionType.BAN


@pytest.mark.asyncio
async def test_executor_timeout_variants(executor):
    """Test that different timeout durations are handled correctly."""
    timeouts = [
        ActionType.TIMEOUT_60S,
        ActionType.TIMEOUT_600S,
    ]
    
    for i, timeout in enumerate(timeouts):
        user_id = f"user_{i}"
        await executor.execute_action(user_id, timeout, f"timeout {i}")
        
        assert executor.user_state[user_id][0] == timeout


@pytest.mark.asyncio
async def test_executor_is_duplicate_logic(executor):
    """Test the _is_duplicate method directly."""
    user_id = "test_user"
    action = ActionType.WARN
    
    # No history - not duplicate
    is_dup = executor._is_duplicate(user_id, action)
    assert not is_dup
    
    # Record action
    executor.user_state[user_id] = (action, time.time())
    
    # Same action within window - is duplicate
    is_dup = executor._is_duplicate(user_id, action)
    assert is_dup
    
    # Different action - not duplicate
    is_dup = executor._is_duplicate(user_id, ActionType.TIMEOUT_60S)
    assert not is_dup


@pytest.mark.asyncio
async def test_executor_window_constants(executor):
    """Test that rate limiting and dedup windows are correct."""
    # Acceptance criteria: 10 commands/second (0.1s minimum interval)
    assert executor.min_interval == 0.1
    
    # Acceptance criteria: 5 second dedup window
    assert executor.dedupe_window == 5.0
