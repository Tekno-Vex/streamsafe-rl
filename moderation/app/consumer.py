"""
Kafka Consumer for Moderation Service
Reads chat messages from Kafka and sends them to the moderation API
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

import httpx
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModerationConsumer:
    """Consumes chat events from Kafka and sends to moderation API"""

    def __init__(
        self,
        kafka_bootstrap: str = "kafka:9092",
        kafka_topic: str = "chat_events",
        moderation_url: str = "http://localhost:8000/moderate"
    ):
        self.kafka_bootstrap = kafka_bootstrap
        self.kafka_topic = kafka_topic
        self.moderation_url = moderation_url
        self.consumer = None
        self.http_client = httpx.AsyncClient(timeout=5.0)

    def start_consumer(self):
        """Initialize Kafka consumer"""
        logger.info(f"Connecting to Kafka at {self.kafka_bootstrap}")
        
        self.consumer = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_bootstrap,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest messages
            enable_auto_commit=True,
            group_id='moderation-consumer-group',
        )
        
        logger.info(f"Successfully subscribed to topic: {self.kafka_topic}")

    async def process_message(self, message: Dict[str, Any]):
        """Send message to moderation API"""
        try:
            # Extract fields from Kafka message
            user_id = message.get("UserID", "unknown")
            username = message.get("User", "unknown")
            channel = message.get("Channel", "unknown")
            content = message.get("Content", "").strip()
            room_id = message.get("RoomID", "unknown")
            
            # Skip empty messages
            if not content:
                return
            
            # Build moderation request
            payload = {
                "message_id": f"{user_id}_{datetime.utcnow().timestamp()}",
                "user_id": user_id,
                "username": username,
                "channel_id": room_id,
                "message_text": content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to moderation API
            response = await self.http_client.post(
                self.moderation_url,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Moderated message from {username}: "
                    f"action={result.get('action')}, "
                    f"risk={result.get('risk_score'):.2f}, "
                    f"latency={result.get('latency_ms'):.1f}ms"
                )
            else:
                logger.error(f"Moderation API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to process message: {e}")

    async def consume_loop(self):
        """Main consumer loop"""
        logger.info("Starting consumer loop...")
        
        for message in self.consumer:
            try:
                await self.process_message(message.value)
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")

    async def run(self):
        """Start the consumer"""
        self.start_consumer()
        await self.consume_loop()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down consumer...")
        if self.consumer:
            self.consumer.close()
        await self.http_client.aclose()
        logger.info("Consumer shutdown complete")


async def main():
    """Main entry point"""
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    kafka_topic = os.getenv("KAFKA_TOPIC", "chat_events")
    moderation_url = os.getenv("MODERATION_URL", "http://localhost:8000/moderate")
    
    consumer = ModerationConsumer(
        kafka_bootstrap=kafka_bootstrap,
        kafka_topic=kafka_topic,
        moderation_url=moderation_url
    )
    
    try:
        await consumer.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await consumer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())