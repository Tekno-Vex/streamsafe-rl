package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

func TestInitRegistersMetrics(t *testing.T) {
	cleanupMetrics()

	Init()

	// Ensure at least one label is used so the CounterVec is exported.
	RateLimitHits.WithLabelValues("test").Inc()

	mfs, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("gather failed: %v", err)
	}

	expected := map[string]bool{
		"ingestion_messages_total":              false,
		"ingestion_messages_dropped_total":      false,
		"ingestion_queue_depth":                 false,
		"ingestion_rate_limited_messages_total": false,
		"ingestion_messages_per_second":         false,
		"ingestion_drops_per_second":            false,
	}

	for _, mf := range mfs {
		if _, ok := expected[mf.GetName()]; ok {
			expected[mf.GetName()] = true
		}
	}

	for name, found := range expected {
		if !found {
			t.Fatalf("expected metric %s to be registered", name)
		}
	}

	cleanupMetrics()
}

func cleanupMetrics() {
	prometheus.Unregister(MessagesProcessed)
	prometheus.Unregister(MessagesDropped)
	prometheus.Unregister(QueueDepth)
	prometheus.Unregister(RateLimitHits)
	prometheus.Unregister(MessagesPerSecond)
	prometheus.Unregister(DropsPerSecond)
}
