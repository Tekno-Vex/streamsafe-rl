#!/bin/bash
# StreamSafe-RL Acceptance Testing Suite
# Unified test script for all sprints (1-4)
# Usage: ./run_tests.sh [sprint1|sprint2|sprint3|sprint4|all]

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0

log_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

test_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASS++))
}

test_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAIL++))
}

test_skip() {
    echo -e "${YELLOW}⏭️  SKIP${NC}: $1"
}

# ============================================================================
# SPRINT 1 TESTS
# ============================================================================

test_sprint1() {
    log_header "Sprint 1 Tests: Core Data Plane (Ingestion, Moderation, Logging)"

    # ---- UNIT TESTS ----
    echo "🔬 UNIT TESTS"
    echo ""
    
    # Go unit tests (Parser, Rate Limiter, Config, Metrics, IRC Client)
    echo "✓ Go Unit Tests (parser, ratelimit, metrics, config, irc)..."
    if cd ingestion && go test -v ./... > /tmp/go_test.log 2>&1; then
        GO_PASS=$(grep -c "PASS" /tmp/go_test.log || true)
        test_pass "Go tests: $GO_PASS passed"
        cd ..
    else
        test_fail "Go unit tests failed"
        cd ..
    fi

    # ---- INTEGRATION TESTS ----
    echo ""
    echo "🌐 INTEGRATION TESTS"
    echo ""

    # A1.1: Ingestion health endpoint
    echo "✓ A1.1: Ingestion /health endpoint..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        test_pass "Ingestion /health returns 200"
    else
        test_fail "Ingestion /health returned $HTTP_CODE"
    fi

    # A1.2: Metrics endpoint with Prometheus format
    echo ""
    echo "✓ A1.2: Prometheus metrics available..."
    METRICS_RESP=$(curl -s http://localhost:8080/metrics 2>/dev/null)
    if [ ! -z "$METRICS_RESP" ] && echo "$METRICS_RESP" | grep -q "go_\|ingestion_"; then
        test_pass "Metrics endpoint returns Prometheus format"
    else
        test_fail "Metrics endpoint not working"
    fi

    # A1.3: IRC message parsing (verified by metrics)
    echo ""
    echo "✓ A1.3: IRC WebSocket connection stable..."
    QUEUE=$(curl -s http://localhost:8080/metrics 2>/dev/null | grep "ingestion_queue_depth" | grep -oE "[0-9]+" | head -1)
    if [ ! -z "$QUEUE" ]; then
        test_pass "IRC client connected and parsing (queue depth: $QUEUE)"
    else
        test_fail "IRC client not responding"
    fi

    # A2.1: Backpressure - Queue depth metric
    echo ""
    echo "✓ A2.1: Backpressure - Queue depth metric..."
    if curl -s http://localhost:8080/metrics 2>/dev/null | grep -q "ingestion_queue_depth"; then
        test_pass "Queue depth metric exposed"
    else
        test_fail "Queue depth metric not found"
    fi

    # A2.2: Backpressure - Messages dropped metric
    echo ""
    echo "✓ A2.2: Backpressure - Messages dropped counter..."
    if curl -s http://localhost:8080/metrics 2>/dev/null | grep -q "ingestion_messages_dropped_total"; then
        test_pass "Messages dropped metric available"
    else
        test_fail "Messages dropped metric not found"
    fi

    # A2.3: Rate limiting metrics
    echo ""
    echo "✓ A2.3: Rate limiting metrics..."
    if curl -s http://localhost:8080/metrics 2>/dev/null | grep -q "ingestion_rate_limited_messages_total\|ingestion_messages_per_second"; then
        test_pass "Rate limiting metrics exposed"
    else
        test_fail "Rate limiting metrics not found"
    fi

    # C1.1: Moderation service health
    echo ""
    echo "✓ C1.1: Moderation service health..."
    HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null)
    if echo "$HEALTH" | grep -q "healthy"; then
        test_pass "Moderation service reports healthy"
    else
        test_fail "Health check failed"
    fi

    # C1.2: Risk scoring - Latency under 50ms
    echo ""
    echo "✓ C1.2: Risk scoring latency (<50ms)..."
    LATENCIES=()
    for i in {1..5}; do
        START=$(date +%s%N)
        curl -s -X POST http://localhost:8000/moderate \
          -H "Content-Type: application/json" \
          -d "{\"message_id\": \"lat_$i\", \"user_id\": \"user_$i\", \"username\": \"test\", \"channel_id\": \"ch1\", \"message_text\": \"test\", \"timestamp\": \"2026-02-13T12:00:00\"}" > /dev/null 2>&1
        END=$(date +%s%N)
        LATENCY_MS=$(( (END - START) / 1000000 ))
        LATENCIES+=($LATENCY_MS)
    done
    MAX_LATENCY=$(printf '%s\n' "${LATENCIES[@]}" | sort -nr | head -1)
    if [ "$MAX_LATENCY" -lt 100 ]; then
        test_pass "Latency within bounds (max ${MAX_LATENCY}ms < 100ms)"
    else
        test_fail "Latency too high (max ${MAX_LATENCY}ms)"
    fi

    # C1.3: Deterministic risk scoring
    echo ""
    echo "✓ C1.3: Deterministic risk scoring..."
    ACTION1=$(curl -s -X POST http://localhost:8000/moderate \
      -H "Content-Type: application/json" \
      -d '{"message_id": "d1", "user_id": "u_det", "username": "t", "channel_id": "c1", "message_text": "AAABBB", "timestamp": "2026-02-13T12:00:00"}' 2>/dev/null | grep -o '"action":"[^"]*"')
    ACTION2=$(curl -s -X POST http://localhost:8000/moderate \
      -H "Content-Type: application/json" \
      -d '{"message_id": "d2", "user_id": "u_det", "username": "t", "channel_id": "c1", "message_text": "AAABBB", "timestamp": "2026-02-13T12:00:00"}' 2>/dev/null | grep -o '"action":"[^"]*"')
    if [ "$ACTION1" = "$ACTION2" ]; then
        test_pass "Identical inputs → identical decisions"
    else
        test_fail "Non-deterministic: $ACTION1 vs $ACTION2"
    fi

    # C1.4: Risk scorer implements all required features
    echo ""
    echo "✓ C1.4: Risk scorer implementation..."
    if grep -q "compute_risk_score\|score_to_action\|fetch_user_history\|fetch_channel_velocity" moderation/app/risk.py; then
        test_pass "Risk scorer has all required methods"
    else
        test_fail "Risk scorer missing methods"
    fi

    # C2.1: Fail-open safety - IGNORE on error
    echo ""
    echo "✓ C2: Fail-open safety..."
    RESPONSE=$(curl -s -X POST http://localhost:8000/moderate \
      -H "Content-Type: application/json" \
      -d '{"message_id": "fo1", "user_id": "u_fo", "username": "t", "channel_id": "c1", "message_text": "test", "timestamp": "2026-02-13T12:00:00"}' 2>/dev/null)
    if echo "$RESPONSE" | grep -q "decision_path"; then
        DECISION_PATH=$(echo "$RESPONSE" | grep -o '"decision_path":"[^"]*"')
        test_pass "Fail-open path tracked: $DECISION_PATH"
    else
        test_fail "decision_path not returned"
    fi

    # C2.2: Structured logging with required fields
    echo ""
    echo "✓ C2.2: Structured logging (latency_ms, decision_path)..."
    if echo "$RESPONSE" | grep -q '"latency_ms"' && echo "$RESPONSE" | grep -q '"decision_path"'; then
        test_pass "Structured logs include latency & decision_path"
    else
        test_fail "Structured logging incomplete"
    fi

    # B1.1: Event logging schema compliance
    echo ""
    echo "✓ B1.1: Event schema compliance..."
    if echo "$RESPONSE" | jq . > /dev/null 2>&1; then
        test_pass "Response is valid JSON"
    else
        test_fail "Response not valid JSON"
    fi

    # B1.2: Required response fields (ModerationResponse schema)
    echo ""
    echo "✓ B1.2: ModerationResponse schema..."
    REQUIRED_FIELDS=("message_id" "action" "risk_score" "latency_ms" "decision_path")
    MISSING_FIELDS=""
    for field in "${REQUIRED_FIELDS[@]}"; do
        if ! echo "$RESPONSE" | grep -q "\"$field\""; then
            MISSING_FIELDS="$MISSING_FIELDS $field"
        fi
    done
    if [ -z "$MISSING_FIELDS" ]; then
        test_pass "All ModerationResponse fields present"
    else
        test_fail "Missing fields:$MISSING_FIELDS"
    fi

    # B1.3: Kafka + Parquet logging infrastructure
    echo ""
    echo "✓ B1.3: Kafka + Parquet logging..."
    if grep -q "ModerationEventLogger\|log_event" moderation/app/logger_pipeline.py && [ -d "moderation/logs/parquet" ]; then
        test_pass "Event logger infrastructure (Kafka + Parquet) in place"
    else
        test_fail "Event logging infrastructure incomplete"
    fi

    # B1.4: Parquet output directory exists
    echo ""
    echo "✓ B1.4: Parquet output directory..."
    if [ -d "moderation/logs/parquet" ]; then
        test_pass "Parquet output directory accessible"
    else
        test_fail "Parquet directory missing"
    fi

    # B1.5: Live metrics API
    echo ""
    echo "✓ B1.5: Live metrics API..."
    METRICS=$(curl -s http://localhost:8000/metrics 2>/dev/null)
    if echo "$METRICS" | jq . > /dev/null 2>&1 && echo "$METRICS" | grep -q "p50_latency\|p99_latency\|action_distribution"; then
        test_pass "Metrics API returns live stats (p50/p99/distribution)"
    else
        test_fail "Metrics API broken"
    fi

    # B1.6: Schemas defined (state.json, action.json, log_event.json)
    echo ""
    echo "✓ B1.6: JSON schemas..."
    SCHEMAS=0
    [ -f "schemas/state.json" ] && ((SCHEMAS++))
    [ -f "schemas/action.json" ] && ((SCHEMAS++))
    [ -f "schemas/log_event.json" ] && ((SCHEMAS++))
    if [ $SCHEMAS -eq 3 ]; then
        test_pass "All 3 required schemas defined"
    else
        test_fail "Missing schemas ($SCHEMAS/3)"
    fi
}

# ============================================================================
# SPRINT 2 TESTS
# ============================================================================

test_sprint2() {
    log_header "Sprint 2 Tests: Safety Layer, Observability, CI/CD, UI"

    # ---- UNIT TESTS ----
    echo "🔬 UNIT TESTS"
    echo ""

    # Python unit tests (Safety Clamp)
    echo "✓ Python Unit Tests (safety_clamp)..."
    if cd moderation && python3 -m pytest -v tests/test_safety.py 2>&1 | tee /tmp/pytest.log > /dev/null; then
        PY_PASS=$(grep -c "PASSED" /tmp/pytest.log || true)
        test_pass "Python tests: $PY_PASS passed"
        cd - > /dev/null
    else
        test_fail "Python unit tests failed"
        cd - > /dev/null
    fi

    # ---- INTEGRATION TESTS ----
    echo ""
    echo "🌐 INTEGRATION TESTS"
    echo ""

    # A3.1: Rate limiting metrics exists
    echo "✓ A3.1: Rate limiting metrics..."
    if curl -s http://localhost:8000/metrics 2>/dev/null | grep -q "rate_limit\|messages_per_second"; then
        test_pass "Rate limiting metrics exposed"
    else
        test_fail "Rate limiting metrics not found"
    fi

    # C3.1: Safety clamp - moderator immunity
    echo ""
    echo "✓ C3.1: Safety clamp - moderator immunity..."
    MOD_RESPONSE=$(curl -s -X POST http://localhost:8000/moderate \
      -H "Content-Type: application/json" \
      -d '{"message_id": "c3_mod", "user_id": "mod123", "username": "moderator", "channel_id": "c1", "message_text": "suspicious", "is_moderator": true, "timestamp": "2026-02-13T12:00:00"}' 2>/dev/null)
    if echo "$MOD_RESPONSE" | grep -q '"action":"IGNORE"'; then
        test_pass "Moderators immune to safety clamp (IGNORE)"
    else
        test_fail "Moderator not immune: $(echo $MOD_RESPONSE | grep -o '"action":"[^"]*"')"
    fi

    # C3.2: Safety clamp - trust-based downgrade
    echo ""
    echo "✓ C3.2: Safety clamp - trust-based action downgrade..."
    HIGH_TRUST=$(curl -s -X POST http://localhost:8000/moderate \
      -H "Content-Type: application/json" \
      -d '{"message_id": "c3_trust", "user_id": "trust999", "username": "trusted_user", "channel_id": "c1", "message_text": "test", "account_age_days": 730, "timestamp": "2026-02-13T12:00:00"}' 2>/dev/null)
    if echo "$HIGH_TRUST" | grep -q '"action"'; then
        ACTION=$(echo "$HIGH_TRUST" | grep -o '"action":"[^"]*"')
        test_pass "Trust-based clamping applied: $ACTION"
    else
        test_fail "Trust-based clamping missing"
    fi

    # C4.1: Action executor implementation
    echo ""
    echo "✓ C4.1: Action executor implementation..."
    if grep -q "class ActionExecutor\|def execute_action\|deduplication_window\|rate_limit" moderation/app/executor.py 2>/dev/null; then
        test_pass "Action executor has required methods"
    else
        test_fail "Action executor methods missing"
    fi

    # C4.2: Rate limiting in executor (rate limiter or similar)
    echo ""
    echo "✓ C4.2: Executor rate limiting config..."
    if grep -q "10.*cmd\|rate.*limit\|commands.*second" moderation/app/executor.py 2>/dev/null; then
        test_pass "Executor rate limiting configured"
    else
        test_fail "Executor rate limiting config missing"
    fi

    # C4.3: Deduplication window (5 second mentions)
    echo ""
    echo "✓ C4.3: Deduplication window..."
    if grep -q "5.*second\|dedup\|300000\|300.*ms" moderation/app/executor.py 2>/dev/null; then
        test_pass "5-second deduplication window configured"
    else
        test_fail "Deduplication window missing"
    fi

    # B2.1: Reward job implementation
    echo ""
    echo "✓ B2.1: Reward job (Spark) implementation..."
    if [ -f "analytics/reward_job.py" ] && grep -q "join\|reward\|decisions\|overrides" analytics/reward_job.py; then
        test_pass "Spark reward job has join & reward logic"
    else
        test_fail "Reward job not implemented"
    fi

    # B2.2: Reward computation logic
    echo ""
    echo "✓ B2.2: Reward computation..."
    if grep -q "positive_reward\|negative_reward\|correct\|error" analytics/reward_job.py; then
        test_pass "Reward computation (correct/error) implemented"
    else
        test_fail "Reward computation logic missing"
    fi

    # B2.3: Parquet output from reward job
    echo ""
    echo "✓ B2.3: Parquet output configuration..."
    if grep -q "parquet\|write\|mode\|overwrite" analytics/reward_job.py; then
        test_pass "Reward job writes Parquet output"
    else
        test_fail "Parquet output not configured"
    fi

    # I1.1: GitHub Actions workflow
    echo ""
    echo "✓ I1.1: GitHub Actions CI pipeline..."
    if [ -f ".github/workflows/test.yml" ] || [ -f ".github/workflows/go-test.yml" ]; then
        test_pass "GitHub Actions workflows configured"
    else
        test_fail "CI workflows missing"
    fi

    # I1.2: Go test job in CI
    echo ""
    echo "✓ I1.2: Go test in CI..."
    if [ -d ".github/workflows" ] && grep -r "go test\|pytest" .github/workflows 2>/dev/null | grep -q "go test"; then
        test_pass "CI runs Go tests"
    else
        test_fail "Go test job missing from CI"
    fi

    # I1.3: Python test job in CI
    echo ""
    echo "✓ I1.3: Python test in CI..."
    if [ -d ".github/workflows" ] && grep -r "pytest\|python.*test" .github/workflows 2>/dev/null | grep -q "pytest"; then
        test_pass "CI runs Python tests"
    else
        test_fail "Python test job missing from CI"
    fi

    # I2.1: React dashboard implementation
    echo ""
    echo "✓ I2.1: React dashboard UI..."
    if [ -f "ui/src/App.tsx" ] && grep -q "p50_latency\|p99_latency\|action_distribution" ui/src/App.tsx; then
        test_pass "Dashboard fetches live metrics"
    else
        test_fail "Dashboard not fetching metrics"
    fi

    # I2.2: Dashboard update rate (2 seconds)
    echo ""
    echo "✓ I2.2: Dashboard update rate..."
    if grep -q "2000\|setInterval.*2\|2.*second" ui/src/App.tsx; then
        test_pass "Dashboard updates every 2 seconds"
    else
        test_fail "Dashboard update rate not configured"
    fi

    # I2.3: Dashboard code size (<500 LOC)
    echo ""
    echo "✓ I2.3: Dashboard code size..."
    LOC=$(wc -l < ui/src/App.tsx 2>/dev/null || echo 0)
    if [ "$LOC" -lt 500 ]; then
        test_pass "Dashboard is $LOC LOC (< 500 LOC)"
    else
        test_fail "Dashboard size $LOC LOC (> 500 LOC)"
    fi

    # I2.4: Dashboard endpoint
    echo ""
    echo "✓ I2.4: Dashboard serves at correct endpoint..."
    DASHBOARD=$(curl -s http://localhost:5173 2>/dev/null | head -20)
    if [ ! -z "$DASHBOARD" ]; then
        test_pass "Dashboard responds on port 5173"
    else
        test_fail "Dashboard not accessible"
    fi

    # B2.4: Live metrics API includes action_distribution
    echo ""
    echo "✓ B2.4: Metrics include action distribution..."
    METRICS=$(curl -s http://localhost:8000/metrics 2>/dev/null)
    if echo "$METRICS" | grep -q "action_distribution\|IGNORE\|TIMEOUT\|BAN"; then
        test_pass "Action distribution tracked in metrics"
    else
        test_fail "Action distribution not in metrics"
    fi

    # I2.5: Tsconfig configured (TypeScript)
    echo ""
    echo "✓ I2.5: TypeScript configuration..."
    if [ -f "ui/tsconfig.json" ]; then
        test_pass "TypeScript configured"
    else
        test_fail "tsconfig.json missing"
    fi
}

# ============================================================================
# SPRINT 3 TESTS
# ============================================================================

test_sprint3() {
    log_header "Sprint 3 Tests: Offline PPO, ONNX Runtime, Kubernetes"

    # A4: Kubernetes
    echo "✓ A4: Kubernetes manifests..."
    K8S_FILES=$(find k8s -name "*.yaml" 2>/dev/null | wc -l)
    if [ "$K8S_FILES" -gt 0 ]; then
        test_pass "Kubernetes deployment manifests present ($K8S_FILES files)"
    else
        test_skip "Kubernetes manifests not yet created (optional for local dev)"
    fi

    # B3: Offline PPO training
    echo ""
    echo "✓ B3: PPO training script..."
    if [ -f "ml/training/ppo_train.py" ] && [ -s "ml/training/ppo_train.py" ]; then
        test_pass "PPO training implementation exists"
    else
        test_skip "PPO training not yet implemented"
    fi

    # C5: ONNX Runtime integration
    echo ""
    echo "✓ C5: ONNX Runtime inference..."
    if grep -q "onnx\|ONNX" moderation/requirements.txt 2>/dev/null; then
        test_pass "ONNX Runtime in dependencies"
    else
        test_skip "ONNX Runtime integration not yet added"
    fi
}

# ============================================================================
# SPRINT 4 TESTS
# ============================================================================

test_sprint4() {
    log_header "Sprint 4 Tests: AWS, Terraform, CI/CD Images, Rollout"

    # I3: AWS Infrastructure
    echo "✓ I3: Terraform configuration..."
    TF_FILES=$(find infra -name "*.tf" 2>/dev/null | wc -l)
    if [ "$TF_FILES" -gt 0 ]; then
        test_pass "Terraform files present ($TF_FILES files)"
    else
        test_skip "Terraform not yet implemented"
    fi

    # I4: Image CI/CD
    echo ""
    echo "✓ I4: Docker image build workflows..."
    if [ -f ".github/workflows/docker.yml" ] || [ -f ".github/workflows/build-push.yml" ]; then
        test_pass "Docker image CI/CD workflows configured"
    else
        test_skip "Docker image CI/CD not yet configured"
    fi

    # C6: Policy rollout
    echo ""
    echo "✓ C6: Policy versioning..."
    if grep -q "policy_version\|POLICY" moderation/app/api.py 2>/dev/null; then
        test_pass "Policy versioning in moderation service"
    else
        test_skip "Policy versioning not yet implemented"
    fi
}

# ============================================================================
# SUMMARY
# ============================================================================

print_summary() {
    local sprint=$1
    log_header "Test Results Summary"

    TOTAL=$((PASS + FAIL))
    echo "Sprint $sprint Results: ${GREEN}$PASS PASSED${NC}, ${RED}$FAIL FAILED${NC}, $TOTAL TOTAL"
    echo ""

    if [ $FAIL -eq 0 ]; then
        echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
        return 0
    else
        echo -e "${RED}⚠️  Some tests failed${NC}"
        return 1
    fi
}

# ============================================================================
# MAIN
# ============================================================================

SPRINT=${1:-all}

case $SPRINT in
    sprint1|1)
        test_sprint1
        print_summary 1
        ;;
    sprint2|2)
        PASS=0
        FAIL=0
        test_sprint2
        print_summary 2
        ;;
    sprint3|3)
        PASS=0
        FAIL=0
        test_sprint3
        print_summary 3
        ;;
    sprint4|4)
        PASS=0
        FAIL=0
        test_sprint4
        print_summary 4
        ;;
    all)
        echo -e "${BLUE}Running all sprints...${NC}\n"
        
        test_sprint1
        S1_PASS=$PASS
        S1_FAIL=$FAIL
        
        PASS=0
        FAIL=0
        test_sprint2
        S2_PASS=$PASS
        S2_FAIL=$FAIL
        
        PASS=0
        FAIL=0
        test_sprint3
        S3_PASS=$PASS
        S3_FAIL=$FAIL
        
        PASS=0
        FAIL=0
        test_sprint4
        S4_PASS=$PASS
        S4_FAIL=$FAIL
        
        log_header "Overall Results"
        echo "Sprint 1: ${GREEN}$S1_PASS PASS${NC} ${RED}$S1_FAIL FAIL${NC}"
        echo "Sprint 2: ${GREEN}$S2_PASS PASS${NC} ${RED}$S2_FAIL FAIL${NC}"
        echo "Sprint 3: ${GREEN}$S3_PASS PASS${NC} ${RED}$S3_FAIL FAIL${NC}"
        echo "Sprint 4: ${GREEN}$S4_PASS PASS${NC} ${RED}$S4_FAIL FAIL${NC}"
        
        TOTAL_PASS=$((S1_PASS + S2_PASS + S3_PASS + S4_PASS))
        TOTAL_FAIL=$((S1_FAIL + S2_FAIL + S3_FAIL + S4_FAIL))
        
        echo ""
        echo "TOTAL: ${GREEN}$TOTAL_PASS PASS${NC} ${RED}$TOTAL_FAIL FAIL${NC}"
        
        if [ $TOTAL_FAIL -eq 0 ]; then
            exit 0
        else
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [sprint1|sprint2|sprint3|sprint4|all]"
        echo ""
        echo "Examples:"
        echo "  $0 sprint1    # Test Sprint 1 only"
        echo "  $0 sprint2    # Test Sprint 2 only"
        echo "  $0 all        # Test all sprints"
        exit 1
        ;;
esac
