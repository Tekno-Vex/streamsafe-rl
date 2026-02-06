import logging
from enum import Enum

# Define our allowed actions
class ActionType(str, Enum):
    IGNORE = "IGNORE"
    WARN = "WARN"
    TIMEOUT_60S = "TIMEOUT_60S"
    TIMEOUT_600S = "TIMEOUT_600S"
    BAN = "BAN"

logger = logging.getLogger("safety_clamp")

class SafetyClamp:
    """
    The Safety Clamp forces actions to stay within safe bounds.
    It downgrades actions if they violate safety policies.
    """
    
    def __init__(self):
        # Configuration: Maximum allowed action based on trust
        self.TRUST_THRESHOLDS = {
            ActionType.BAN: 0.9,
            ActionType.TIMEOUT_600S: 0.7,
            ActionType.TIMEOUT_60S: 0.5
        }

    def clamp(self, requested_action: str, user_trust_score: float, is_moderator: bool) -> str:
        """
        Takes a requested action and returns a safe action.
        """
        final_action = requested_action

        # RULE 1: NEVER ban or timeout a Moderator
        # If they are a mod, the only allowed action is IGNORE (do nothing)
        if is_moderator and requested_action != ActionType.IGNORE:
            logger.warning(f"CLAMP: Prevented action {requested_action} against MODERATOR")
            return ActionType.IGNORE

        # RULE 2: Trust-based clamping
        # If a user is trusted (score > 0.8), don't BAN them immediately.
        if requested_action == ActionType.BAN and user_trust_score > 0.8:
             logger.info(f"CLAMP: Downgraded BAN to TIMEOUT_600S due to high trust")
             final_action = ActionType.TIMEOUT_600S

        return final_action