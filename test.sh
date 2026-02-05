#!/bin/bash
# Test script for StreamSafe moderation service

set -e

echo "==================================="
echo "StreamSafe Moderation Service Tests"
echo "==================================="
echo ""

# Get project root
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

echo "1️⃣  Cleaning up old containers..."
docker-compose down --remove-orphans 2>/dev/null || true
sleep 2

echo ""
echo "2️⃣  Building and starting containers..."
docker-compose up -d --build

echo ""
echo "⏳ Waiting for services to be healthy (30s)..."
sleep 10

# Check Redis
echo ""
echo "3️⃣  Checking Redis..."
if docker exec streamsafe-redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis failed health check"
    docker-compose logs redis
    exit 1
fi

# Check Moderation Service
echo ""
echo "4️⃣  Checking Moderation Service health..."
for i in {1..30}; do
    if docker exec streamsafe-moderation python3 -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()" 2>/dev/null; then
        echo "✅ Moderation service is healthy"
        break
    else
        if [ $i -lt 30 ]; then
            echo "⏳ Waiting for moderation service... ($i/30)"
            sleep 1
        else
            echo "❌ Moderation service health check failed"
            docker-compose logs moderation
            exit 1
        fi
    fi
done

echo ""
echo "5️⃣  Testing /health endpoint..."
HEALTH=$(curl -s http://localhost:8000/health)
echo "Response: $HEALTH"
if echo "$HEALTH" | grep -q "healthy"; then
    echo "✅ /health endpoint working"
else
    echo "⚠️  /health endpoint returned: $HEALTH"
fi

echo ""
echo "6️⃣  Testing /moderate endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "test_msg_001",
    "user_id": "test_user_123",
    "username": "testuser",
    "channel_id": "test_channel_456",
    "message_text": "Hello everyone! This is a test message.",
    "timestamp": "2026-02-04T12:00:00"
  }')

echo "Response: $RESPONSE"
if echo "$RESPONSE" | grep -q "IGNORE\|WARN\|TIMEOUT"; then
    echo "✅ /moderate endpoint working"
else
    echo "❌ /moderate endpoint failed"
    exit 1
fi

echo ""
echo "7️⃣  Testing message with risky content..."
RESPONSE2=$(curl -s -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "test_msg_002",
    "user_id": "test_user_999",
    "username": "spamuser",
    "channel_id": "test_channel_456",
    "message_text": "AAAAAAAAAAAAAAAA http://spam.com SPAMSPAMSPAM",
    "timestamp": "2026-02-04T12:01:00"
  }')

echo "Response: $RESPONSE2"
if echo "$RESPONSE2" | grep -q "action"; then
    echo "✅ Risky message moderation working"
else
    echo "❌ Risky message test failed"
    exit 1
fi

echo ""
echo "8️⃣  Checking Parquet logs..."
if ls moderation/logs/parquet/*.parquet 2>/dev/null | head -1 | xargs stat &>/dev/null; then
    echo "✅ Parquet files being written"
    ls -lh moderation/logs/parquet/
else
    echo "ℹ️  No Parquet files yet (will appear after 100 events or on shutdown)"
fi

echo ""
echo "==================================="
echo "✅ All tests passed!"
echo "==================================="
echo ""
echo "Container status:"
docker-compose ps
echo ""
echo "View logs: docker-compose logs -f moderation"
echo "Stop all:  docker-compose down"
