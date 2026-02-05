# HTTP service that will expose the risk scorer to handle incoming messages
# Purpose: FastAPI service that receives moderation requests, uses RiskScorer to compute risk scores, and enforces fail-open safety policy.

import asyncio
import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from app.logger_pipeline import ModerationEventLogger

import uvloop
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis.asyncio as redis

from app.risk import RiskScorer, ActionType

# use uvloop for better async performance (target of p99)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",)
logger = logging.getLogger(__name__)

## Pydantic Models for request and response validation

class ModerationRequest(BaseModel):
    """Request schema for moderation API"""
    message_id: str = Field(..., description="Unique message identifier")
    user_id: str = Field(..., description="Twitch user ID")
    username: str = Field(..., description="Twitch username")
    channel_id: str = Field(..., description="Twitch channel ID")
    message_text: str = Field(..., description="Chat message content", max_length=500)
    timestamp: str = Field(..., description="ISO 8601 timestamp of the message sent")

class ModerationResponse(BaseModel):
    """Response schema for moderation API"""
    message_id: str
    action: str # ActionType enum
    risk_score: float
    latency_ms: float
    decision_path: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    redis_connected: bool
    timestamp: str

## FastAPI app setup
# Globals for connection management
redis_client: redis.Redis = None
risk_scorer: RiskScorer = None
event_logger: ModerationEventLogger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asnc context manager for app startup and shutdown
    
    Startup: Initialize Redis client and RiskScorer instance
    Shutdown: Cleanup connections
    """
    global redis_client, risk_scorer, event_logger

    # Startup
    logger.info("Starting moderation service...")
    
    try:
        # Initialize Redis client
        # In local development, we connect to localhost:6379
        # In Docker, use redis://redis:6379
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = await redis.from_url(redis_url, decode_responses=True)

        # Test Redis connection
        await redis_client.ping()
        logger.info("Redis connected successfully")

        # Initialize RiskScorer
        risk_scorer = RiskScorer(redis_client=redis_client, redis_timeout=0.5)
        logger.info("RiskScorer initialized successfully")

        # Initialize Event Logger
        event_logger = ModerationEventLogger(kafka_enabled=False, parquet_output_dir="./logs/parquet", parquet_batch_size=100,)
        logger.info("Event Logger initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

    yield # app runs here

    ## Shutdown
    logger.info("Shutting down moderation service...")
    if event_logger:
        await event_logger.shutdown()
    if redis_client:
        await redis_client.close()
    logger.info("Service shutdown complete")

# Create FastAPI app with lifespan context manager
app = FastAPI(title="StreamSafe-RL Moderation Service", description="Async moderation decisions with fail-open safety", version="1.0.0", lifespan=lifespan,)

## Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint

    Returns:
        - status: "healthy" if all dependencies are ready, "degraded" otherwise
        - redis_connected: whether Redis is reachable
    """
    redis_ok = False
    if redis_client:
        try:
            await asyncio.wait_for(redis_client.ping(), timeout=1.0)
            redis_ok = True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            redis_ok = False
    
    status = "healthy" if redis_ok else "degraded"

    return HealthResponse(status=status, redis_connected=redis_ok, timestamp=datetime.utcnow().isoformat(),)

@app.post("/moderate", response_model=ModerationResponse)
async def moderate(request: ModerationRequest) -> ModerationResponse:
    """
    Main moderation endpoint.

    Recieves a chat message and returns a deterministic moderation decision.

    Args:
        request: ModerationRequest with message details
    
    Returns:
        ModerationResponse with action, risk score, and latency

    Raises:
        HTTPException: Only on invalid input (not on dependency failures - those fail-open)
    """
    import time
    start_time = time.time()

    try:
        logger.info(f"Received moderation request: user={request.message_id}, "
                    f"channel={request.channel_id}, message_id={request.message_id}")

        # validate request
        if not request.message_text or len(request.message_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="message text cannot be empty")
        
        # call risk scorer (with detailed feature tracking for logging)
        try:
            # Track redis fetch time separately
            redis_start = time.time()
            user_history, channel_velocity = await asyncio.gather(
                risk_scorer.fetch_user_history(request.user_id, request.channel_id),
                risk_scorer.fetch_channel_velocity(request.channel_id),
            )
            redis_fetch_ms = (time.time() - redis_start) * 1000.0

            # Extract message features (no redis call)
            message_features = risk_scorer.extract_message_features(request.message_text)

            # Compute risk score
            risk_score = risk_scorer.compute_risk_score(
                user_history, channel_velocity, message_features
            )

            # Map to action
            action = risk_scorer.score_to_action(risk_score)
            decision_path = "deterministic_risk_only"
            failure_reason = None
        
        except asyncio.TimeoutError:
            logger.warning(f"Risk scoring timeout for message {request.message_id}  failing open")
            action = ActionType.IGNORE
            risk_score = 0.0
            decision_path = "fail-open-timeout"
            failure_reason = "Risk scoring timeout"
            redis_fetch_ms = 0.0
            # Use empty defaults for logging
            user_history = {"messages_last_hour": 0, "warnings_last_24h": 0, "timeouts_last_7d": 0, "account_age_days": 0, "is_subscriber": False, "is_moderator": False}
            channel_velocity = {"messages_per_minute": 0.0, "active_users": 0, "recent_timeouts": 0}
            message_features = {"contains_caps": False, "contains_urls": False, "repeated_chars": False}

        except Exception as e:
            logger.error(f"Error during risk scoring {e}", exc_info=True)
            action = ActionType.IGNORE
            risk_score = 0.0
            decision_path = "fail-open-error"
            failure_reason = str(e)
            redis_fetch_ms = 0.0
            # Use empty defaults
            user_history = {"messages_last_hour": 0, "warnings_last_24h": 0, "timeouts_last_7d": 0, "account_age_days": 0, "is_subscriber": False, "is_moderator": False}
            channel_velocity = {"messages_per_minute": 0.0, "active_users": 0, "recent_timeouts": 0}
            message_features = {"contains_caps": False, "contains_urls": False, "repeated_chars": False}

        # calculate latency
        latency_ms = (time.time() - start_time) * 1000.0

        # log event for RL training (async, non-blocking)
        asyncio.create_task(
            event_logger.log_event(
                message_id=request.message_id,
                channel_id=request.channel_id,
                user_id=request.user_id,
                username=request.username,
                message_text=request.message_text,
                user_history=user_history,
                channel_velocity=channel_velocity,
                message_features=message_features,
                action_requested=action.value,
                action_final=action.value,
                risk_score=risk_score,
                latency_ms=latency_ms,
                redis_fetch_ms=redis_fetch_ms,
                decision_path=decision_path,
                failure_reason=failure_reason,
            )
        )

        logger.info(f"Moderation decision: message_id={request.message_id}, action={action}, risk={risk_score:.3f}, latency={latency_ms:.1f}ms")

        return ModerationResponse(
            message_id=request.message_id,
            action=action.value,
            risk_score=risk_score,
            latency_ms=latency_ms,
            decision_path=decision_path,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /moderate: {e}", exc_info=True)
        latency_ms = (time.time() - start_time) * 1000.0
        return ModerationResponse(
            message_id=request.message_id,
            action=ActionType.IGNORE.value,
            risk_score=0.0,
            latency_ms=latency_ms,
            decision_path="fail_open_error",
        )

@app.post("/log_event")
async def log_event(event: Dict[str, Any]) -> Dict[str, str]:
    """
    Log an event for offline RL training.

    This endpoint will be used later to publish events to Kafka/Redis streams.
    For now, it just acknowledges receipt.

    Args:
        event: Arbitrary event data as a dictionary matching log_event.json schema.
    Returns:
        Confirmation with event_id.
    """
    event_id = str(uuid.uuid4())
    logger.info(f"Logged event {event_id}")
    return {"event_id": event_id, "status": "received"}

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint for basic service info.
    """
    return {"service": "StreamSafe-RL Moderation Service", "version": "1.0.0", "endpoints": ["/health", "/moderate", "/log_event"],}

## Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # run with uvloop and multiple workers for production
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info", access_log=True,)