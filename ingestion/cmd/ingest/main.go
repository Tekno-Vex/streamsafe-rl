package main

import (
	"log"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/config"
	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/irc"
	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/metrics" // Import this
	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/ratelimit"
	"github.com/prometheus/client_golang/prometheus/promhttp" // Import this
)

func main() {
	log.Println("Starting StreamSafe Ingestion Service (Engineer 1)...")

	cfg := config.LoadConfig()

	// Initialize Metrics
	metrics.Init()

	// Initialize Rate Limiter (2000 messages per second per channel)
	limiter := ratelimit.NewPerChannelLimiter(2000)

	// counters for throughput calculation
	var (
		processedCount int64
		droppedCount   int64
	)

	go func() {
		// Existing Health Check
		http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("OK"))
		})

		// NEW: Prometheus Metrics Endpoint
		http.Handle("/metrics", promhttp.Handler())

		log.Println("Health & Metrics running on :8080")
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	client := irc.NewClient(cfg.TwitchURL, cfg.TwitchChannel)

	go func() {
		// calculate and update throughput metrics every second
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()

		var lastProcessed, lastDropped int64
		lastTime := time.Now()

		for range ticker.C {
			now := time.Now()
			elapsed := now.Sub(lastTime).Seconds()
			lastTime = now

			currentProcessed := atomic.LoadInt64(&processedCount)
			currentDropped := atomic.LoadInt64(&droppedCount)

			mps := float64(currentProcessed-lastProcessed) / elapsed
			dps := float64(currentDropped-lastDropped) / elapsed

			metrics.MessagesPerSecond.Set(mps)
			metrics.DropsPerSecond.Set(dps)

			lastProcessed = currentProcessed
			lastDropped = currentDropped
		}
	}()

	go func() {
		for msg := range client.DataChan {
			// Rate limiting check
			if !limiter.Allow(msg.Channel) {
				atomic.AddInt64(&droppedCount, 1)
				metrics.MessagesDropped.Inc()
				metrics.RateLimitHits.WithLabelValues(msg.Channel).Inc()
				continue
			}

			atomic.AddInt64(&processedCount, 1)
			metrics.MessagesProcessed.Inc()
			metrics.QueueDepth.Set(float64(len(client.DataChan)))

			// process message (send to moderation service, log, etc.)
			_ = msg
		}
	}()

	client.Connect()
}
