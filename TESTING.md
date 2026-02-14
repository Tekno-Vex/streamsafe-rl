# Testing Guide

## Overview

All acceptance testing is consolidated into a single test script: **`run_tests.sh`**

This unified approach eliminates test file sprawl and provides a consistent interface for testing across all sprints.

## Quick Start

### Run all tests (all sprints)
```bash
./run_tests.sh all
```

### Run specific sprint tests
```bash
./run_tests.sh sprint1    # Sprint 1: Core Data Plane
./run_tests.sh sprint2    # Sprint 2: Safety & Observability
./run_tests.sh sprint3    # Sprint 3: Offline PPO & Kubernetes
./run_tests.sh sprint4    # Sprint 4: AWS & Production Hardening
```

### Shorthand
```bash
./run_tests.sh 1          # Same as sprint1
./run_tests.sh 2          # Same as sprint2
```

## Test Structure

The `run_tests.sh` script contains **4 test functions**, one per sprint:

### Sprint 1: Core Data Plane (11 tests)
- **A1.1**: Ingestion `/health` endpoint response time
- **A1.2**: Prometheus metrics format
- **A2.1-A2.2**: Queue depth & backpressure metrics
- **C1.1-C1.3**: Moderation health, latency, determinism
- **C2**: Fail-open safety (decision path tracking)
- **B1**: Event logging schema & Parquet output

### Sprint 2: Safety & Observability (7 tests)
- **A3**: Rate limiting metrics
- **C3**: Safety clamp (moderator protection, trust-based clamping)
- **C4**: Action executor (deduplication, rate limiting)
- **B2**: Spark reward calculation (joining, rewards)
- **I2**: React dashboard (<500 LOC, live metrics)
- **I1**: GitHub Actions CI pipeline

### Sprint 3: Offline PPO & Kubernetes (3 checks)
- **A4**: Kubernetes manifests
- **B3**: PPO training script (optional - skipped if not implemented)
- **C5**: ONNX Runtime integration (optional - skipped)

### Sprint 4: AWS & Production Hardening (3 checks)
- **I3**: Terraform infrastructure files
- **I4**: Docker build/push CI/CD workflows
- **C6**: Policy versioning in moderation service

## Test Results

Each test shows:
- ✅ **PASS** - Requirement met
- ❌ **FAIL** - Requirement not met (critical)
- ⏭️  **SKIP** - Feature not yet implemented (optional)

### Example Output
```
Sprint 1 Results: 11 PASSED, 0 FAILED, 11 TOTAL
Sprint 2 Results: 7 PASSED, 0 FAILED, 7 TOTAL
Sprint 3 Results: 1 PASSED, 0 FAILED, 2 SKIPPED
Sprint 4 Results: 3 PASSED, 0 FAILED, 3 TOTAL

TOTAL: 22 PASS, 0 FAIL
```

## Adding New Tests

To add a test for Sprint N:

1. Create a `test_sprintN()` function in `run_tests.sh`:
```bash
test_sprintN() {
    log_header "Sprint N Tests: Description"
    
    echo "✓ Feature name..."
    if [ condition ]; then
        test_pass "Description"
    else
        test_fail "Error description"
    fi
}
```

2. Add a case in the main menu:
```bash
case $SPRINT in
    sprintN|N)
        PASS=0
        FAIL=0
        test_sprintN
        print_summary N
        ;;
```

## Running Tests in CI/CD

In GitHub Actions:
```yaml
- name: Run Tests
  run: ./run_tests.sh all
```

Exit codes:
- **0** - All tests passed
- **1** - Some tests failed (should block merge)

## Services Required

Tests assume services are running:
- **Ingestion service**: http://localhost:8080 (`/health`, `/metrics`)
- **Moderation service**: http://localhost:8000 (`/health`, `/metrics`, `/moderate`)
- **Redis**: localhost:6379 (used by moderation)

To start services:
```bash
docker-compose up -d
```

## Debugging Tests

To see detailed output, add `set -x` at the top of `run_tests.sh`:
```bash
set -x
bash run_tests.sh sprint1
```

Or run individual curl commands manually:
```bash
# Check ingestion health
curl -v http://localhost:8080/health

# Check metrics
curl http://localhost:8080/metrics | grep ingestion_

# Test moderation
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"message_id": "test", "user_id": "user1", "username": "test", "channel_id": "channel1", "message_text": "hello", "timestamp": "2026-02-13T12:00:00"}'
```

## Test Coverage by Story

| User Story | Test | Sprint |
|-----------|------|--------|
| A1: IRC Ingestion | Health/metrics endpoints | 1 |
| A2: Backpressure | Queue depth metrics | 1 |
| A3: Rate Limiting | Rate limit metrics | 2 |
| A4: Kubernetes | K8s manifests present | 3 |
| A5: Production Hardening | (planned) | 4 |
| C1: Risk Scoring | Latency, determinism | 1 |
| C2: Fail-Open | decision_path tracking | 1 |
| C3: Safety Clamp | Clamp implementation | 2 |
| C4: Action Executor | Dedup & rate limiting | 2 |
| C5: ONNX Integration | ONNX dependencies | 3 |
| C6: Policy Rollout | Policy versioning | 4 |
| B1: Logging | Schema + Parquet | 1 |
| B2: Rewards | Spark reward job | 2 |
| B3: PPO Training | ppo_train.py exists | 3 |
| B4: Evaluation | (planned) | 4 |
| I1: CI Pipeline | GitHub Actions | 2 |
| I2: Dashboard UI | React metrics display | 2 |
| I3: AWS Infrastructure | Terraform files | 4 |
| I4: Image CI/CD | Docker workflows | 4 |

