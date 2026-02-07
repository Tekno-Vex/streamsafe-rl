from app.safety_clamp import SafetyClamp
import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum

import redis.asyncio as redis 

# configure logging
logger = logging.getLogger(__name__)

class ActionType(str, Enum):
    """Action types matching action.json schema"""
    IGNORE = "IGNORE"
    WARN = "WARN"
    TIMEOUT_60S = "TIMEOUT_60S"
    TIMEOUT_600S = "TIMEOUT_600S"
    TIMEOUT_86400S = "TIMEOUT_86400S"
    BAN = "BAN"

class RiskScorer:
    """
    Deterministic risk scorer for moderation decisions.

    Combines user history (from Redis) and message features
    into a risk score (0.0 to 1.0), then maps to discrete actions.
    """

    # Risk thresholds for deterministic action mapping
    THRESHOLD_WARN = 0.4 # Risk >=0.4 and < 0.7 -> WARN
    THRESHOLD_TIMEOUT_60S = 0.7 # Risk >=0.7 and < 0.85 -> TIMEOUT_60S
    THRESHOLD_TIMEOUT_600S = 0.85 # Risk >=0.85 and < 0.95 -> TIMEOUT_600S
    THRESHOLD_BAN = 0.95 # Risk >=0.95 -> BAN

    # Feature weights for weighted scoring
    WEIGHTS = {
        "warnings_last_24h": 0.2,
        "timeouts_last_7d": 0.25,
        "messages_last_hour": 0.15,
        "messages_per_minute": 0.1,
        "repeated_chars": 0.1,
        "contains_urls": 0.1,
        "contains_caps": 0.05,
        "is_moderator": -0.1, # negative = reduces risk
        "is_subscriber": -0.05 # negative = reduces risk
    }

    def __init__(self, redis_client: redis.Redis, redis_timeout: float = 0.5):
        """ Initialize risk scorer """
        """ Args:
            redis_client: An instance of an async Redis client.
            redis_timeout: Maximum time for Redis operations in seconds.
        """
        self.redis_client = redis_client
        self.redis_timeout = redis_timeout
        self.safety = SafetyClamp()
    
    async def fetch_user_history(self, user_id: str, channel_id: str) -> Dict[str, Any]:
        """ Fetch user behavior history from Redis """
        
        """ Redis key structure (you can adjust this):
            - {channel_id}:user:{user_id}:messages_last_hour → int
            - {channel_id}:user:{user_id}:warnings_24h → int
            - {channel_id}:user:{user_id}:timeouts_7d → int
            - user:{user_id}:account_age_days → int
            - {channel_id}:user:{user_id}:is_subscriber → 0/1
            - {channel_id}:user:{user_id}:is_moderator → 0/1
        
            On Redis timeout or error, returns safe defaults (all zeros/false).
        """
        try: 
            # Fetch all user features with timeout
            pipeline = self.redis_client.pipeline()
            pipeline.get(f"{channel_id}:user:{user_id}:messages_last_hour")
            pipeline.get(f"{channel_id}:user:{user_id}:warnings_24h")
            pipeline.get(f"{channel_id}:user:{user_id}:timeouts_7d")
            pipeline.get(f"user:{user_id}:account_age_days")
            pipeline.get(f"{channel_id}:user:{user_id}:is_subscriber")
            pipeline.get(f"{channel_id}:user:{user_id}:is_moderator")

            results = await asyncio.wait_for(pipeline.execute(), timeout=self.redis_timeout)

            history = {"messages_last_hour": int(results[0] or 0),
                       "warnings_last_24h": int(results[1] or 0),
                       "timeouts_last_7d": int(results[2] or 0),
                       "account_age_days": int(results[3] or 0),
                       "is_subscriber": bool(int(results[4] or 0)),
                       "is_moderator": bool(int(results[5] or 0)),
            }
            logger.info(f"Fetched user history for {user_id}: {history}")
            return history
        
        except asyncio.TimeoutError:
            logger.warning(f"Redis timeout fetching history for {user_id} using defaults")
            return {
                "messages_last_hour": 0,
                "warnings_last_24h": 0,
                "timeouts_last_7d": 0,
                "account_age_days": 0,
                "is_subscriber": False,
                "is_moderator": False,
            }
        
        except Exception as e:
            logger.error(f"Error fetching history for {user_id}: {e}")
            return {
                "messages_last_hour": 0,
                "warnings_last_24h": 0,
                "timeouts_last_7d": 0,
                "account_age_days": 0,
                "is_subscriber": False,
                "is_moderator": False,
            }
        
    async def fetch_channel_velocity(self, channel_id: str) -> Dict[str, Any]:
        """
        Fetch channel-level activity metrics for Redis.
        
        Redis key structure:
            - {channel_id}:messages_per_minute → float
            - {channel_id}:active_users -> int
            - {channel_id}:recent_timeouts -> int

        On Redis timeout or error, returns safe defaults.
        """

        try:
            pipeline = self.redis_client.pipeline()
            pipeline.get(f"{channel_id}:messages_per_minute")
            pipeline.get(f"{channel_id}:active_users")
            pipeline.get(f"{channel_id}:recent_timeouts")

            results = await asyncio.wait_for(pipeline.execute(), timeout=self.redis_timeout)

            velocity = {
                "messages_per_minute": float(results[0] or 0.0),
                "active_users": int(results[1] or 0),
                "recent_timeouts": int(results[2] or 0),
            }
            logger.info(f"Fetched channel velocity for {channel_id}: {velocity}")
            return velocity
        
        except asyncio.TimeoutError:
            logger.warning(f"Redis timeout fetching velocity for {channel_id} using defaults")
            return {
                "messages_per_minute": 0.0,
                "active_users": 0,
                "recent_timeouts": 0,
            }
        
        except Exception as e:
            logger.error(f"Error fetching velocity for {channel_id}: {e}")
            return {
                "messages_per_minute": 0.0,
                "active_users": 0,
                "recent_timeouts": 0,
            }
        
    def extract_message_features(self, message_text: str) -> Dict[str, bool]:
        """
        Extract boolean features from message content.
        
        Args:
            message_text: Raw chat message
        
        Returns:
            Dictionary of message features
        """
        features = {
            "contains_caps": self._has_excessive_caps(message_text),
            "contains_urls": self._has_urls(message_text),
            "repeated_chars": self._has_repeated_chars(message_text),
        }
        logger.debug(f"Extracted message features: {features}")
        return features
    
    @staticmethod
    def _has_excessive_caps(text: str) -> bool:
        """ Check if text has excessive capital letters (>50%) """
        if len(text) < 3:
            return False
        caps_count = sum(1 for c in text if c.isupper())
        return (caps_count / len(text)) > 0.5

    @staticmethod
    def _has_urls(text: str) -> bool:
        """ Check if text contains URLs """
        url_indicators = ["http://", "https://", "www.", ".com"]
        return any(indicator in text for indicator in url_indicators)
    
    @staticmethod
    def _has_repeated_chars(text: str) -> bool:
        """Check for repeated character patterns (e.g., 'aaaaaaa')"""
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                return True
        return False
    
    def compute_risk_score(self, user_history: Dict[str, Any], channel_velocity: Dict[str, Any], message_features: Dict[str, bool],) -> float:
        """
        Compute weighted risk score from all features.
        
        Use WEIGHTS dict to combine features into scalar score [0.0, 1.0].
        
        Args:
            user_history: User behavior metrics
            channel_velocity: Channel activity metrics
            message_features: Message content features
        
        Returns:
            Risk score between 0.0 (safe) and 1.0 (definitely ban)
        """
        score = 0.0

        # Normalize user history features to [0.0, 1.0]
        # Warnnings: assume max 5 in 24h -> normalize
        warnings_risk = min(user_history["warnings_last_24h"] / 5.0, 1.0)
        score += warnings_risk * self.WEIGHTS["warnings_last_24h"]

        # Timeouts: assume max 3 in 7d -> normalize
        timeouts_risk = min(user_history["timeouts_last_7d"] / 3.0, 1.0)
        score += timeouts_risk * self.WEIGHTS["timeouts_last_7d"]

        # Message spam: assume max 100 messages in 1h is max risk
        messages_risk = min(user_history["messages_last_hour"] / 100.0, 1.0)
        score += messages_risk * self.WEIGHTS["messages_last_hour"]

        # Channel velocity: assume max 100 messages/min is risky
        velocity_risk = min(channel_velocity["messages_per_minute"] / 100.0, 1.0)
        score += velocity_risk * self.WEIGHTS["messages_per_minute"]

        # Message features (binary so directly weight)
        if message_features.get("repeated_chars"):
            score += self.WEIGHTS["repeated_chars"]
        
        if message_features.get("contains_urls"):
            score += self.WEIGHTS["contains_urls"]
        
        if message_features.get("contains_caps"):
            score += self.WEIGHTS["contains_caps"]
        
        # User status (reduce risk)
        if user_history.get("is_moderator"):
            score += self.WEIGHTS["is_moderator"] # negative weight
        
        if user_history.get("is_subscriber"):
            score += self.WEIGHTS["is_subscriber"] # negative weight
        
        # Clamp score to [0.0, 1.0]
        risk_score = max(0.0, min(score, 1.0))
        logger.info(f"Computed risk score: {risk_score:.3f}")

        return risk_score
    
    def score_to_action(self, risk_score: float) -> ActionType:
        """
        Deterministically map risk score to action.
        
        This is competely deterministic: same risk_score always produces same action.
        
        Args:
            risk_score: Score from 0.0 to 1.0
        
        Returns:
            ActionType (IGNORE, WARN, TIMEOUT_60S, TIMEOUT_600S, TIMEOUT_86400S, or BAN)"""
        if risk_score >= self.THRESHOLD_BAN:
            return ActionType.BAN
        elif risk_score >= self.THRESHOLD_TIMEOUT_600S:
            return ActionType.TIMEOUT_600S
        elif risk_score >= self.THRESHOLD_TIMEOUT_60S:
            return ActionType.TIMEOUT_60S
        elif risk_score >= self.THRESHOLD_WARN:
            return ActionType.WARN
        else:
            return ActionType.IGNORE
        
    async def score_messages(self, user_id: str, channel_id: str, message_text: str) -> tuple[ActionType, float, str]:
        """
        Main entry point: score a message and return action.
        
        Args:
            user_id: Twitch user ID
            channel_id: Twitch channel ID
            message_text: Chat message content
            
        Returns:
            Tuple of (action_type, risk_score, decision_path)
            decision_path indicates which branch was taken (for debugging)
        """
        try:
            # fetch user and channel data in parallel
            user_history, channel_velocity = await asyncio.gather(self.fetch_user_history(user_id, channel_id), self.fetch_channel_velocity(channel_id),)

            # extract message features
            message_features = self.extract_message_features(message_text)
            
            # compute risk score
            risk_score = self.compute_risk_score(user_history, channel_velocity, message_features)

            # map to action
            action = self.score_to_action(risk_score)

            # --- START SAFETY CLAMP ---
            # 1. Get safety features (Default to safe values if missing)
            is_mod = features.get("is_moderator", 0) == 1
            trust_score = 0.5  # Placeholder until we have real trust data

            # 2. Clamp the action
            safe_action = self.safety.clamp(action, trust_score, is_mod)

            # 3. Update the action variable
            action = safe_action

            logger.info(f"Scored message from {user_id}: risk={risk_score:.3f}, action={action}")
            
            return action, risk_score, "deterministic_path"

        except Exception as e:
            logger.error(f"Error scoring message: {e}", exc_info=True)
            # Fail-open: default to IGNORE on any error
            return ActionType.IGNORE, 0.0, "fail_open_error"
        
        