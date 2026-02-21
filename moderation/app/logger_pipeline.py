# Purpose: Module that publishes moderation events to Kafka and writes to Parquet files

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)

class ModerationEventLogger:
    """
    Logs moderation events for offline RL training.

    Publishes to:
    - Kafka topic for real-time streaming
    - Parquet files for batch training
    """
    
    SCHEMA_VERSION = "1.0.0"
    KAFKA_TOPIC = "moderation-events"

    # Parquet schema matching log_event.json
    PARQUET_SCHEMA = pa.schema([
        ('event_id', pa.string()),
        ('schema_version', pa.string()),
        ('timestamp', pa.timestamp('us')),

        # state fields
        ('message_id', pa.string()),
        ('channel_id', pa.string()),
        ('user_id', pa.string()),
        ('username', pa.string()),
        ("message_text", pa.string()),
        ("message_length", pa.int32()),
        
        # User history
        ("user_messages_last_hour", pa.int32()),
        ("user_warnings_last_24h", pa.int32()),
        ("user_timeouts_last_7d", pa.int32()),
        ("user_account_age_days", pa.int32()),
        ("user_is_subscriber", pa.bool_()),
        ("user_is_moderator", pa.bool_()),
        
        # Channel velocity
        ("channel_messages_per_minute", pa.float64()),
        ("channel_active_users", pa.int32()),
        ("channel_recent_timeouts", pa.int32()),
        
        # Message features
        ("feature_contains_caps", pa.bool_()),
        ("feature_contains_urls", pa.bool_()),
        ("feature_repeated_chars", pa.bool_()),
        
        # Actions
        ("action_requested", pa.string()),
        ("action_requested_confidence", pa.float64()),
        ("action_requested_source", pa.string()),
        ("action_final", pa.string()),
        ("action_final_confidence", pa.float64()),
        ("action_final_source", pa.string()),
        ("was_clamped", pa.bool_()),
        
        # Latency breakdown
        ("latency_total_ms", pa.float64()),
        ("latency_redis_fetch_ms", pa.float64()),
        ("latency_risk_scoring_ms", pa.float64()),
        ("latency_rl_inference_ms", pa.float64()),
        ("latency_safety_clamp_ms", pa.float64()),
        
        # RL Shadow Mode Data
        ("rl_action", pa.string()),
        ("rl_probs", pa.string()),  # JSON string of probabilities
        ("rl_latency_ms", pa.float64()),
        ("rl_agreement", pa.bool_()),
        
        # Decision metadata
        ("decision_path", pa.string()),
        ("failure_reason", pa.string()),
        ("risk_score", pa.float64()),
    ])

    def __init__(self, kafka_enabled: bool = True, kafka_bootstrap_servers: str = "localhost:9092", parquet_output_dir: str = "./logs/parquet", parquet_batch_size: int = 1000,):
        """
        Initialize event logger.
        
        Args:
            kafka_enabled: Whether to publish to Kafka
            Kafka_bootstrap_servers: Kafka broker addresses
            parquet_output_dir: Directory to store Parquet files
            parquet_batch_size: Number of events per Parquet file
        """
        self.kafka_enabled = kafka_enabled
        self.parquet_output_dir = Path(parquet_output_dir)
        self.parquet_batch_size = parquet_batch_size

        # create output dir
        self.parquet_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kafka producer
        self.kafka_producer: Optional[KafkaProducer] = None
        if self.kafka_enabled:
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks=1,  # wait for leader acknowledgment
                    retries=3,
                    max_in_flight_requests_per_connection=5,
                )
                logger.info(f"Kafka producer initialized: {kafka_bootstrap_servers}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka producer: {e}")
                self.kafka_producer = None
        
        # In-memory buffer for Parquet batch writing
        self.parquet_buffer = []
        self.parquet_lock = asyncio.Lock()

    async def log_event(
        self,
        message_id: str,
        channel_id: str,
        user_id: str,
        username: str,
        message_text: str,
        user_history: Dict[str, Any],
        channel_velocity: Dict[str, Any],
        message_features: Dict[str, bool],
        action_requested: str,
        action_final: str,
        risk_score: float,
        latency_ms: float,
        redis_fetch_ms: float,
        decision_path: str,
        failure_reason: Optional[str] = None,
        rl_action: Optional[str] = None,
        rl_probs: Optional[Dict[str, float]] = None,
        rl_latency_ms: float = 0.0,
        rl_agreement: bool = False,
    ) -> str:
        """
        Log a moderation event.
        Returns:
            event_id: Unique identifier for the logged event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # build complete event matching log_event.json schema
        event = {
            "event_id": event_id,
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": timestamp.isoformat(),

            # State
            "state": {
                "message_id": message_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "username": username,
                "message_text": message_text,
                "message_length": len(message_text),
                "timestamp": timestamp.isoformat(),
                "user_history": user_history,
                "channel_velocity": channel_velocity,
                "features": message_features,
            },

            # Actions
            "action_requested": {
                "action_type": action_requested,
                "confidence": risk_score,
                "source": "risk_scorer",
                "timestamp": timestamp.isoformat(),
            },
            "action_final": {
                "action_type": action_final,
                "confidence": risk_score,
                "source": "risk_scorer", # Later: "safety_clamp" if clamped
                "timestamp": timestamp.isoformat(),
            },
            "was_clamped": False,

            # Latency
            "latency_ms": {
                "total": latency_ms,
                "redis_fetch": redis_fetch_ms,
                "risk_scoring": latency_ms - redis_fetch_ms,
                "rl_inference": rl_latency_ms,
                "safety_clamp": 0.0,
            },

            # RL Shadow Mode Data
            "rl_shadow": {
                "action": rl_action,
                "action_probs": rl_probs or {},
                "latency_ms": rl_latency_ms,
                "agreement_with_baseline": rl_agreement,
            },

            # Metadata
            "decision_path": decision_path,
            "failure_reason": failure_reason,
            "metadata": {
                "service_version": "1.0.0",
                "deployment_env": "local",
            },
        }

        # Publish to Kafka
        if self.kafka_producer:
            try:
                future = self.kafka_producer.send(self.KAFKA_TOPIC, value=event)
                # don't wait for the acknowledgment (fire-and-forget for low latency)
                logger.debug(f"Published event {event_id} to Kafka")
            except KafkaError as e:
                logger.warning(f"Failed to publish to Kafka: {e}")
        
        # add to parquet buffer
        await self._buffer_for_parquet(event, user_history, channel_velocity, message_features)

        logger.info(f"Logged event {event_id}: {action_final} for {message_id}")
        return event_id

    async def _buffer_for_parquet(
        self,
        event: Dict[str, Any],
        user_history: Dict[str, Any],
        channel_velocity: Dict[str, Any],
        message_features: Dict[str, bool],
    ):
        """
        Add event to parquet buffer and write if batch size reached.
        """
        async with self.parquet_lock:
            # Flatten even to match parquet schema
            flattened = {
                "event_id": event["event_id"],
                "schema_version": event["schema_version"],
                "timestamp": datetime.fromisoformat(event["timestamp"]),

                # state fields
                "message_id": event["state"]["message_id"],
                "channel_id": event["state"]["channel_id"],
                "user_id": event["state"]["user_id"],
                "username": event["state"]["username"],
                "message_text": event["state"]["message_text"],
                "message_length": event["state"]["message_length"],

                # User history
                "user_messages_last_hour": user_history.get("messages_last_hour", 0),
                "user_warnings_last_24h": user_history.get("warnings_last_24h", 0),
                "user_timeouts_last_7d": user_history.get("timeouts_last_7d", 0),
                "user_account_age_days": user_history.get("account_age_days", 0),
                "user_is_subscriber": user_history.get("is_subscriber", False),
                "user_is_moderator": user_history.get("is_moderator", False),

                # Channel velocity
                "channel_messages_per_minute": channel_velocity.get("messages_per_minute", 0.0),
                "channel_active_users": channel_velocity.get("active_users", 0),
                "channel_recent_timeouts": channel_velocity.get("recent_timeouts", 0),

                # Message features
                "feature_contains_caps": message_features.get("contains_caps", False),
                "feature_contains_urls": message_features.get("contains_urls", False),
                "feature_repeated_chars": message_features.get("repeated_chars", False),

                # Actions
                "action_requested": event["action_requested"]["action_type"],
                "action_requested_confidence": event["action_requested"]["confidence"],
                "action_requested_source": event["action_requested"]["source"],
                "action_final": event["action_final"]["action_type"],
                "action_final_confidence": event["action_final"]["confidence"],
                "action_final_source": event["action_final"]["source"],
                "was_clamped": event["was_clamped"],

                # Latency
                "latency_total_ms": event["latency_ms"]["total"],
                "latency_redis_fetch_ms": event["latency_ms"]["redis_fetch"],
                "latency_risk_scoring_ms": event["latency_ms"]["risk_scoring"],
                "latency_rl_inference_ms": event["latency_ms"]["rl_inference"],
                "latency_safety_clamp_ms": event["latency_ms"]["safety_clamp"],

                # RL Shadow Mode Data
                "rl_action": event["rl_shadow"].get("action", ""),
                "rl_probs": json.dumps(event["rl_shadow"].get("action_probs", {})),
                "rl_latency_ms": event["rl_shadow"].get("latency_ms", 0.0),
                "rl_agreement": event["rl_shadow"].get("agreement_with_baseline", False),

                # Metadata
                "decision_path": event["decision_path"],
                "failure_reason": event.get("failure_reason", ""),
                "risk_score": event["action_requested"]["confidence"],
            }

            self.parquet_buffer.append(flattened)

            # Write batch if buffer full
            if len(self.parquet_buffer) >= self.parquet_batch_size:
                await self._flush_parquet_buffer()

    async def _flush_parquet_buffer(self):
        """
        Write buffered events to parquet file.
        """
        if not self.parquet_buffer:
            return
        
        try:
            # create filename with timestamp
            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = self.parquet_output_dir / f"events_{timestamp_str}.parquet"
            
            # Convert buffer to PyArrow Table
            table = pa.Table.from_pylist(self.parquet_buffer, schema=self.PARQUET_SCHEMA)

            # Write to Parquet file (appends if exists)
            pq.write_table(table, filename, compression='snappy')

            logger.info(
                f"Wrote {len(self.parquet_buffer)} events to {filename}"
            )

            # Clear buffer
            self.parquet_buffer.clear()
        
        except Exception as e:
            logger.error(f"Failed to write Parquet file: {e}", exc_info=True)

    async def shutdown(self):
        """
        Flush remaining events and close connections.
        """
        logger.info("Shutting down event logger...")

        # Flush remaining parquet events
        await self._flush_parquet_buffer()

        # Close Kafka producer
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        
        logger.info("Event logger shutdown complete")
