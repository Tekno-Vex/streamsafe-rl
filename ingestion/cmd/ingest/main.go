package main

import (
	"log"
	"net/http"

	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/config"
	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/irc"
	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/metrics" // Import this
	"github.com/prometheus/client_golang/prometheus/promhttp"       // Import this
)

func main() {
	log.Println("Starting StreamSafe Ingestion Service (Engineer 1)...")

	cfg := config.LoadConfig()

	// Initialize Metrics
	metrics.Init()

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
		for msg := range client.DataChan {
			// Just printing for now, effectively "consuming" the queue
			_ = msg
			// fmt.Printf("[%s] %s: %s\n", msg.Channel, msg.User, msg.Content)
			// (Commented out print to test speed, feel free to uncomment to see chat)
		}
	}()

	client.Connect()
}
