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
)

// Register metrics with Prometheus
func Init() {
	prometheus.MustRegister(MessagesProcessed)
	prometheus.MustRegister(MessagesDropped)
	prometheus.MustRegister(QueueDepth)
}
