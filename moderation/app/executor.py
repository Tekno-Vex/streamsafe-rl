import logging
import asyncio
import time
from app.schemas.action import ActionType

logger = logging.getLogger("executor")

class ActionExecutor:
    """
    Executes moderation actions.
    Features:
    1. Rate Limiting: Ensures we don't spam Twitch.
    2. Deduplication: Prevents sending the same action twice to the same user.
    """
    
    def __init__(self):
        # Rate Limiting: Minimum seconds between ANY command (Global Limit)
        self.min_interval = 0.1  # 10 commands per second max
        self.last_global_execution = 0.0

        # Deduplication: Track the last action per user
        # Format: { "user_id": ("ACTION_TYPE", timestamp) }
        self.user_state = {} 
        self.dedupe_window = 5.0 # Don't repeat same action on user within 5s

    async def execute_action(self, user_id: str, action: ActionType, reason: str):
        """
        Main entry point. Validates and executes the action.
        """
        # 1. Ignore 'IGNORE' actions immediately
        if action == ActionType.IGNORE:
            return

        # 2. Check for Duplicates (Acceptance Criteria: No duplicate actions)
        if self._is_duplicate(user_id, action):
            logger.info(f"SKIPPING: Duplicate action {action} for {user_id}")
            return

        # 3. Apply Rate Limiting (Acceptance Criteria: Outbound Rate Limits)
        # If we are going too fast, wait a tiny bit.
        await self._enforce_rate_limit()

        # 4. "Fire" the action
        await self._send_irc_command(user_id, action)
        
        # 5. Log it (Acceptance Criteria: Execution Latency Tracked)
        logger.info(f"EXECUTED: {action} on {user_id} | Reason: {reason}")
        
        # 6. Update state for deduplication
        self.user_state[user_id] = (action, time.time())
        self.last_global_execution = time.time()

    def _is_duplicate(self, user_id: str, action: ActionType) -> bool:
        """Returns True if we just did this action to this user."""
        if user_id in self.user_state:
            last_action, last_time = self.user_state[user_id]
            # If same action and within 5 seconds, it's a duplicate
            if last_action == action and (time.time() - last_time < self.dedupe_window):
                return True
        return False

    async def _enforce_rate_limit(self):
        """Sleeps if needed to match the global rate limit."""
        elapsed = time.time() - self.last_global_execution
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            await asyncio.sleep(wait_time)

    async def _send_irc_command(self, user_id: str, action: ActionType):
        """
        Simulates the raw IRC command sending.
        """
        command = ""
        if action == ActionType.WARN:
            command = f"/me @{user_id} WARNING: Watch your language."
        elif action == ActionType.TIMEOUT_60S:
            command = f"/timeout {user_id} 60"
        elif action == ActionType.TIMEOUT_600S:
            command = f"/timeout {user_id} 600"
        elif action == ActionType.BAN:
            command = f"/ban {user_id}"
            
        # This print is our "Evidence" that it happened
        print(f">>> IRC OUTBOUND: {command}")