StreamSafe Ingestion Metrics (Sprint 2)

This document defines the Prometheus metrics exposed by the ingestion service.

Metrics

- ingestion_messages_total (counter)
  - Total number of chat messages processed after rate limiting.
- ingestion_messages_dropped_total (counter)
  - Total number of messages dropped due to rate limiting.
- ingestion_queue_depth (gauge)
  - Current number of messages in the processing channel.
- ingestion_rate_limited_messages_total (counter, labels: channel_id)
  - Total number of messages rejected due to per-channel rate limiting.
- ingestion_messages_per_second (gauge)
  - Current throughput of processed messages per second.
- ingestion_drops_per_second (gauge)
  - Current drop rate of messages per second.

Scrape endpoint

- GET /metrics

Notes

- ingestion_messages_dropped_total increments for rate-limited messages.
- ingestion_rate_limited_messages_total increments only when per-channel rate limiting rejects a message.
