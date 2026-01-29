package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	log.Println("Starting StreamSafe Ingestion Service...")

	// Basic environment check (Simulating a real startup)
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("Ingestion service ready on port %s\n", port)
}