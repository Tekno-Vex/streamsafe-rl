package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
	// Metric 1: Counter for total messages processed
	MessagesProcessed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "ingestion_messages_total",
		Help: "Total number of chat messages processed",
	})

	// Metric 2: Counter for dropped messages (Backpressure)
	MessagesDropped = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "ingestion_messages_dropped_total",
		Help: "Total number of messages dropped due to full queue",
	})

	// Metric 3: Gauge for current Queue Depth
	QueueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "ingestion_queue_depth",
		Help: "Current number of messages in the processing channel",
	})

	// Metric 4: Counter for rate-limited messages
	RateLimitHits = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "ingestion_rate_limited_messages_total",
		Help: "Total number of messages rejected due to rate limiting per channel",
	}, []string{"channel_id"})

	// Metric 5: Gauge for messages per second (throughput)
	MessagesPerSecond = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "ingestion_messages_per_second",
		Help: "Current throughput of messages processed per second",
	})

	// Metric 6: Gauge for dropped messages per second
	DropsPerSecond = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "ingestion_drops_per_second",
		Help: "Current drop rate in messages per second",
	})
)

// Register metrics with Prometheus
func Init() {
	prometheus.MustRegister(MessagesProcessed)
	prometheus.MustRegister(MessagesDropped)
	prometheus.MustRegister(QueueDepth)
	prometheus.MustRegister(RateLimitHits)
	prometheus.MustRegister(MessagesPerSecond)
	prometheus.MustRegister(DropsPerSecond)
}
