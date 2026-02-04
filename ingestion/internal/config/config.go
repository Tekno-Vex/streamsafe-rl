package config

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	TwitchURL     string
	TwitchChannel string
	BufferSize    int
}

func LoadConfig() *Config {
	// Load .env file if it exists (for local dev)
	err := godotenv.Load()
	if err != nil {
		// It's okay if .env doesn't exist (e.g. in Docker), so just log it softly
		log.Println("Note: No .env file found, using system environment variables")
	}

	// Default to Twitch's secure WebSocket
	url := os.Getenv("TWITCH_IRC_URL")
	if url == "" {
		url = "wss://irc-ws.chat.twitch.tv:443"
	}

	// Default to a busy channel for testing
	channel := os.Getenv("TWITCH_CHANNEL")
	if channel == "" {
		channel = "kaicenat"
	}

	return &Config{
		TwitchURL:     url,
		TwitchChannel: channel,
		BufferSize:    1024,
	}
}
