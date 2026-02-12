package config

import "testing"

func TestLoadConfigDefaults(t *testing.T) {
	t.Setenv("TWITCH_IRC_URL", "")
	t.Setenv("TWITCH_CHANNEL", "")

	cfg := LoadConfig()

	if cfg.TwitchURL != "wss://irc-ws.chat.twitch.tv:443" {
		t.Fatalf("expected default TwitchURL, got %q", cfg.TwitchURL)
	}
	if cfg.TwitchChannel != "kaicenat" {
		t.Fatalf("expected default TwitchChannel, got %q", cfg.TwitchChannel)
	}
	if cfg.BufferSize != 1024 {
		t.Fatalf("expected default BufferSize 1024, got %d", cfg.BufferSize)
	}
}

func TestLoadConfigCustomValues(t *testing.T) {
	t.Setenv("TWITCH_IRC_URL", "wss://custom.twitch.tv:8080")
	t.Setenv("TWITCH_CHANNEL", "testchannel")

	cfg := LoadConfig()

	if cfg.TwitchURL != "wss://custom.twitch.tv:8080" {
		t.Fatalf("expected custom TwitchURL, got %q", cfg.TwitchURL)
	}
	if cfg.TwitchChannel != "testchannel" {
		t.Fatalf("expected custom TwitchChannel, got %q", cfg.TwitchChannel)
	}
}

func TestLoadConfigEmptyEnvDefaults(t *testing.T) {
	t.Setenv("TWITCH_IRC_URL", "")
	t.Setenv("TWITCH_CHANNEL", "")

	cfg := LoadConfig()

	if cfg.TwitchURL == "" || cfg.TwitchChannel == "" {
		t.Fatalf("expected defaults when env vars are empty")
	}
}
