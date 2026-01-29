# StreamSafe-RL

A low-latency, safety-critical chat moderation system using deterministic rules and an offline-trained PPO policy under strict latency and safety constraints.

## Architecture
- **Ingestion:** Go service (IRC, backpressure, rate limiting)
- **Decision:** Python service (risk scoring, safety clamps, RL inference)
- **ML:** Offline PPO training with logged experience
- **Safety:** Shadow deployment and rollback guarantees

## Non-Goals
- No online RL (safety risk)
- No live exploration
- No unsafe ML overrides

## Latency Budgets
- Ingestion → Decision: < 20ms
- Decision service p99: < 50ms
- RL inference: < 3ms