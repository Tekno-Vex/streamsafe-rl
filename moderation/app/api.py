# HTTP service that will expose the risk scorer to handle incoming messages
# Purpose: FastAPI service that receives moderation requests, uses RiskScorer to compute risk scores, and enforces fail-open safety policy.

import asyncio
import logging
import time
import statistics
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from collections import deque

# Fast API & Pydantic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis

# App Modules
from .executor import ActionExecutor
from .logger_pipeline import ModerationEventLogger
from .risk import RiskScorer, ActionType
from .onnx_infer import ONNXInferenceEngine

# use uvloop for better async performance (target of p99), NOT ON WINDOWS
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("🚀 uvloop enabled for high-performance async")
except ImportError:
    print("⚠️ uvloop not available (running on Windows?), using standard asyncio")

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
executor: ActionExecutor = None
onnx_engine: ONNXInferenceEngine = None
# --- NEW: METRICS STORE ---
# We keep the last 1000 request times to calculate live p50/p99 latency
latency_history = deque(maxlen=1000)
# We count actions since startup
action_counts: Dict[str, int] = {"BAN": 0, "TIMEOUT_600S": 0, "TIMEOUT_60S": 0, "WARN": 0, "IGNORE": 0}

def update_metrics(action: ActionType, latency_ms: float):
    """Helper to update stats safely"""
    latency_history.append(latency_ms)
    # Convert Enum to string key (e.g., ActionType.BAN -> "BAN")
    key = action.value
    if key in action_counts:
        action_counts[key] += 1
    else:
        action_counts[key] = 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asnc context manager for app startup and shutdown
    
    Startup: Initialize Redis client and RiskScorer instance
    Shutdown: Cleanup connections
    """
    global redis_client, risk_scorer, event_logger, executor, onnx_engine

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
        kafka_enabled_env = os.getenv("KAFKA_ENABLED", "true").strip().lower()
        kafka_enabled = kafka_enabled_env in ("1", "true", "yes", "y")
        kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        parquet_output_dir = os.getenv("PARQUET_OUTPUT_DIR", "./logs/parquet")
        parquet_batch_size = int(os.getenv("PARQUET_BATCH_SIZE", "100"))

        event_logger = ModerationEventLogger(
            kafka_enabled=kafka_enabled,
            kafka_bootstrap_servers=kafka_bootstrap,
            parquet_output_dir=parquet_output_dir,
            parquet_batch_size=parquet_batch_size,
        )
        logger.info("Event Logger initialized successfully")

        # Initialize ActionExecutor
        executor = ActionExecutor() # <--- ADD THIS BLOCK
        logger.info("ActionExecutor initialized successfully")

        # Initialize ONNX Runtime (shadow mode by default)
        onnx_model_path = os.getenv("ONNX_MODEL_PATH", "models/ppo_policy.onnx")
        shadow_mode = os.getenv("SHADOW_MODE", "true").strip().lower() in ("1", "true", "yes", "y")
        
        if os.path.exists(onnx_model_path):
            onnx_engine = ONNXInferenceEngine.load_from_path(onnx_model_path, shadow_mode=shadow_mode)
            logger.info(f"ONNX Runtime initialized (shadow_mode={shadow_mode})")
        else:
            logger.warning(f"ONNX model not found at {onnx_model_path}, RL inference disabled")
            onnx_engine = None

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

# --- NEW: ENABLE CORS ---
# This allows your React app (running on localhost:5173) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

## Endpoints
@app.get("/metrics")
async def get_metrics():
    """Return live system stats for the dashboard"""
    if not latency_history:
        p50 = 0
        p99 = 0
    else:
        p50 = statistics.median(latency_history)
        # Calculate 99th percentile (approximate)
        if len(latency_history) > 1:
            p99 = statistics.quantiles(latency_history, n=100)[98]
        else:
            p99 = latency_history[0]

    return {
        "p50_latency_ms": round(p50, 2),
        "p99_latency_ms": round(p99, 2),
        "action_distribution": action_counts,
        "policy_version": "v1.0.0 (Rule-Based)"
    }

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

            # --- START EXECUTION TRIGGER (User Story C4) ---
            # Fire-and-forget execution task
            # We use create_task so we don't slow down the response waiting for rate limits
            if executor:
                asyncio.create_task(
                    executor.execute_action(
                        user_id=request.user_id,
                        action=action,
                        reason=f"Risk Score: {risk_score:.2f}"
                    )
                )
        
        except asyncio.TimeoutError:
            logger.warning(f"Risk scoring timeout for message {request.message_id}  failing open")
            action = ActionType.IGNORE
            risk_score = 0.0
            decision_path = "fail_open_redis_timeout"
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
            decision_path = "fail_open_error"
            failure_reason = str(e)
            redis_fetch_ms = 0.0
            # Use empty defaults
            user_history = {"messages_last_hour": 0, "warnings_last_24h": 0, "timeouts_last_7d": 0, "account_age_days": 0, "is_subscriber": False, "is_moderator": False}
            channel_velocity = {"messages_per_minute": 0.0, "active_users": 0, "recent_timeouts": 0}
            message_features = {"contains_caps": False, "contains_urls": False, "repeated_chars": False}

        # calculate latency
        latency_ms = (time.time() - start_time) * 1000.0
        update_metrics(action, latency_ms)

        # --- SHADOW MODE RL INFERENCE ---
        rl_action = None
        rl_probs = None
        rl_latency_ms = 0.0
        rl_agreement = False

        if onnx_engine:
            try:
                rl_start = time.time()
                # Construct state features (10-dim) matching ONNX engine expectations
                state_dict = {
                    "risk_score": float(risk_score),
                    "message_count_24h": float(user_history.get("messages_last_hour", 0)),
                    "warning_count": float(user_history.get("warnings_last_24h", 0)),
                    "timeout_count": float(user_history.get("timeouts_last_7d", 0)),
                    "account_age_days": float(user_history.get("account_age_days", 0)),
                    "follower_count": 0.0,  # Not available in current risk score
                    "subscriber": user_history.get("is_subscriber", False),
                    "moderator": user_history.get("is_moderator", False),
                    "channel_velocity": float(channel_velocity.get("messages_per_minute", 0.0)),
                    "trust_score": 1.0 - float(risk_score),  # Inverse of risk
                }
                
                # Get RL prediction (shadow mode - no action taken)
                rl_result = onnx_engine.infer(state_dict)
                rl_latency_ms = (time.time() - rl_start) * 1000.0
                
                if rl_result:
                    rl_action = rl_result.get("action")
                    rl_probs = rl_result.get("action_probs")
                    # Check if RL agrees with baseline decision
                    rl_agreement = (rl_action == action.value)
                    logger.info(f"RL inference: action={rl_action}, agreement={rl_agreement}, latency={rl_latency_ms:.2f}ms")
            except Exception as e:
                logger.warning(f"RL inference failed (shadow mode continues): {e}")
                rl_action = None
                rl_probs = None
                rl_latency_ms = 0.0
                rl_agreement = False

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
                rl_action=rl_action,
                rl_probs=rl_probs,
                rl_latency_ms=rl_latency_ms,
                rl_agreement=rl_agreement,
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
        update_metrics(ActionType.IGNORE, latency_ms)
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