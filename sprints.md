Here is the full, authoritative 2-Engineer Sprint Plan.

I have re-mapped every single user story from the original 3-person plan to Engineer 1 (Systems) and Engineer 2 (ML/Backend). This is the exact schedule you will execute.

🧭 StreamSafe‑RL — FINAL Sprint Plan (2-Engineer Model)
👥 Team Roles
Engineer 1 (Systems & Platform): Owns Go ingestion, Docker, Kubernetes, AWS, Terraform, and CI/CD.

Engineer 2 (Backend & ML): Owns Python services, Offline RL (PPO/PyTorch), Spark, ONNX, and React UI.

🗓 Week 1 — Core Data Plane (Dockerized, Async, Deterministic)
🎯 Sprint Goal: Messages flow from Twitch → Go → Python → Redis → Kafka/Streams with deterministic decisions under strict latency bounds, fully containerized.

👤 Engineer 1 — Systems & Platform
User Story A1: Persistent Twitch IRC Ingestion (Go, WebSockets, Docker) As a backend systems engineer using Go, I want to build a persistent Twitch IRC WebSocket ingestion service so that high‑volume chat traffic can be ingested reliably during burst events without message loss.

Tasks

Implement Go WebSocket client connecting to wss://irc-ws.chat.twitch.tv

Perform IRC capability negotiation using CAP REQ

Implement reconnect logic with exponential backoff + jitter

Parse raw IRC frames at the byte level (no UTF‑8 decoding)

Extract {user_id, room_id, message_bytes, timestamp}

Parameterize configuration via environment variables

Containerize the service using Docker

Acceptance Criteria

Sustains 2,000+ msgs/sec inside Docker

No crashes or message loss during forced reconnects

Memory usage remains bounded over 10+ minutes

/health endpoint responds <100ms under load

User Story A2: Backpressure & Early Filtering (Go, Channels, Prometheus) As a Go systems engineer, I want to apply early filtering and bounded backpressure using Go channels so downstream services are protected during traffic spikes.

Tasks

Drop IRC PING, JOIN, PART messages before full parsing

Implement bounded buffered channels between pipeline stages

Block producers when downstream queues are full

Expose queue depth, drops/sec via Prometheus /metrics

Acceptance Criteria

Queue depth never grows unbounded

Backpressure behavior visible via metrics

System degrades gracefully under simulated raids

👤 Engineer 2 — Backend & ML
User Story C1: Deterministic Risk Scoring (Python, asyncio/uvloop, Redis, Docker) As a backend engineer building an async Python service, I want to compute deterministic moderation risk scores using Redis‑backed features so decisions meet strict p99 latency guarantees.

Tasks

Build async service using FastAPI + asyncio + uvloop

Fetch user history + channel velocity from Redis

Implement logistic / weighted risk scoring function

Define deterministic thresholds for WARN and TIMEOUT

Enforce strict Redis timeouts (fail‑fast)

Containerize service using Docker

Acceptance Criteria

p99 latency <50ms in docker‑compose

Identical inputs produce identical decisions

Redis timeouts do not block request completion

/health endpoint reports readiness

User Story C2: Fail‑Open Safety (Python, Timeouts, Structured Logging) As a reliability‑focused backend engineer, I want the moderation service to fail open when dependencies are unavailable to prevent cascading failures.

Tasks

Implement timeout handling for Redis + inference calls

Default to IGNORE on dependency failure

Emit structured JSON logs: latency_ms, failure_reason, decision_path

Acceptance Criteria

No crashes when Redis is unavailable

Messages continue flowing

Fail‑open behavior observable in logs and metrics

User Story B1: RL‑Ready Logging (Kafka or Redis Streams, Parquet, Schemas) As an ML engineer preparing offline PPO training, I want to log structured moderation decisions so reinforcement learning datasets can be reconstructed reliably.

Tasks

Define and freeze schemas: schemas/state.json, schemas/action.json, schemas/log_event.json

Log (state, action_requested, action_final, latency, timestamp, schema_version)

Publish logs to Kafka or Redis Streams

Sink logs to Parquet (Docker‑mounted volume)

Acceptance Criteria

Schema versioning enforced

Logs replayable into training dataset

No schema ambiguity

🗓 Week 2 — Safety, Observability, CI/CD, UI Foundations
🎯 Sprint Goal: System is safe, observable, continuously tested, and visible via a lightweight React dashboard.

👤 Engineer 1 — Systems & Platform
User Story A3: Rate Limiting & Metrics (Go, Prometheus) As a Go backend engineer, I want to enforce per‑channel rate limiting and expose metrics so ingestion remains stable under raid‑level traffic.

Tasks

Implement per‑channel token‑bucket rate limiting

Expose metrics: msgs/sec, drops/sec, rate‑limit hits

Document metrics contract

Acceptance Criteria

No overload during simulated raids

Metrics scrapeable via Prometheus

User Story I1: CI Pipeline (GitHub Actions, Docker) As a team using GitHub, we want automated CI pipelines so all services are tested consistently.

Tasks

GitHub Actions PR workflow: go test ./..., pytest

Block merge on failure

Acceptance Criteria

CI runs on every PR

Green checks required to merge

👤 Engineer 2 — Backend & ML
User Story C3: Safety Clamp Layer (Python, Centralized Policy) As a backend engineer responsible for safety, I want a centralized clamp layer so neither heuristics nor RL can emit unsafe actions.

Tasks

Enforce min/max action bounds

Enforce trust‑based caps

Enforce latency‑based fallback

Log clamp decisions (requested_action → final_action)

Acceptance Criteria

Unsafe actions never executed

Clamp logic is auditable and unit‑tested

User Story C4: Action Execution (Python, IRC, Rate Limiting) As a backend engineer, I want to execute moderation actions safely and observably.

Tasks

Implement action executor abstraction

Apply rate limiting to outbound IRC commands

Fire‑and‑forget execution logging

Acceptance Criteria

No duplicate actions

Execution latency tracked

User Story B2: Reward Construction (Spark, Parquet) As an ML engineer, I want to compute defensible rewards using Spark so offline PPO training reflects real moderation outcomes.

Tasks

Spark job joins: decision logs + moderator override events

Compute per‑decision reward

Persist PPO training dataset in Parquet

Acceptance Criteria

Reward distribution bounded and explainable

Dataset reproducible from same inputs

User Story I2: Observability UI (React, TypeScript) As a developer operating the system, I want a lightweight React/TypeScript UI to observe system behavior in real time.

Tasks

React + TypeScript app

Read‑only dashboard showing: p50/p99 latency, action distribution, current policy version

Backend exposes /metrics/api endpoint

Acceptance Criteria

UI renders live data

No auth, no write paths

<500 LOC frontend

🗓 Week 3 — Offline PPO, ONNX Runtime, Kubernetes
🎯 Sprint Goal: Real PPO is trained offline, deployed via ONNX Runtime, and the full system runs in local Kubernetes.

👤 Engineer 1 — Systems & Platform
User Story A4: Kubernetes Deployment (Docker, K8s) As a platform‑minded engineer, I want to deploy services to local Kubernetes for orchestration.

Tasks

Write Deployment + Service manifests

Use ConfigMaps for configuration

Deploy via kind/minikube

Acceptance Criteria

System runs end‑to‑end in Kubernetes

No Helm, no cloud dependency

👤 Engineer 2 — Backend & ML
User Story B3: Offline PPO Training (PyTorch, Docker, ONNX) As an ML engineer, I want to train an offline PPO policy using logged data and export it for production inference.

Tasks

Define discrete action space

Train PPO using PyTorch on logged dataset

Evaluate on held‑out window

Export trained policy to ONNX

Acceptance Criteria

Policy improves at least one metric vs baseline

No safety violations in evaluation

ONNX artifact versioned

User Story C5: ONNX Runtime Integration (Python, ONNX Runtime) As a backend engineer, I want to integrate ONNX Runtime so RL inference is low‑latency and safe.

Tasks

Load ONNX model from mounted volume or S3

Implement inference wrapper

Enable shadow mode (no action execution)

Log RL vs deterministic decisions

Acceptance Criteria

Inference latency <3ms

Shadow metrics produced without side effects

🗓 Week 4 — AWS, Terraform, CI/CD Images, Rollout
🎯 Sprint Goal: Cloud artifacts exist, CI builds images automatically, and ML rollout + rollback is demoable.

👤 Engineer 1 — Systems & Platform
User Story I3: AWS Infrastructure (Terraform, S3, ECR) As an engineer using infrastructure‑as‑code, I want to provision minimal AWS resources for artifacts.

Tasks

Terraform configs for: S3 (logs + models), ECR (Go + Python images)

Acceptance Criteria

terraform apply works from clean state

Resources tracked in state

User Story I4: Image CI/CD (GitHub Actions, ECR) As a DevOps‑aware engineer, I want Docker images built and pushed automatically.

Tasks

Build images on merge to main

Push to ECR with version tags

Acceptance Criteria

No manual image builds

CI logs visible

User Story A5: Production Hardening (Go, OS Signals) As a systems engineer, I want ingestion to behave like production software.

Tasks

Graceful shutdown on SIGTERM

Resource cleanup

Final performance benchmark

Acceptance Criteria

Sustains 2k+ msgs/sec

Clean exits and restarts

👤 Engineer 2 — Backend & ML
User Story C6: Safe Policy Rollout & Rollback (Config‑Driven) As a backend engineer, I want safe ML rollout with rollback support.

Tasks

Policy versioning via config

% rollout flag (or shadow vs active toggle)

Rollback via config change

Acceptance Criteria

Rollout auditable

Rollback demoable live

User Story B4: Final Evaluation (ML Metrics) As an ML engineer, I want defensible evaluation artifacts suitable for interviews.

Tasks

FP/FN proxy analysis

Action distribution drift analysis

Write evaluation report

Acceptance Criteria

Metrics match resume claims

No unexplained regressions