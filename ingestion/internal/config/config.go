package config

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	TwitchURL        string
	TwitchChannel    string
	TwitchUsername   string
	TwitchOAuthToken string
	BufferSize       int
}

func LoadConfig() *Config {
	err := godotenv.Load()
	if err != nil {
		log.Println("Note: No .env file found, using system environment variables")
	}

	url := os.Getenv("TWITCH_IRC_URL")
	if url == "" {
		url = "wss://irc-ws.chat.twitch.tv:443"
	}

	channel := os.Getenv("TWITCH_CHANNEL")
	if channel == "" {
		channel = "kaicenat"
	}

	username := os.Getenv("TWITCH_USERNAME")
	if username == "" {
		username = "justinfan12345"
	}

	token := os.Getenv("TWITCH_OAUTH_TOKEN")

	return &Config{
		TwitchURL:        url,
		TwitchChannel:    channel,
		TwitchUsername:   username,
		TwitchOAuthToken: token,
		BufferSize:       1024,
	}
}
