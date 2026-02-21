package irc

import "testing"

func TestNewClientDefaults(t *testing.T) {
	// Update: Pass dummy username and token for the test
	client := NewClient("wss://example.com", "test", "test_user", "oauth:12345")

	if client == nil {
		t.Fatal("expected client")
	}
	if client.url != "wss://example.com" {
		t.Fatalf("expected url to be set")
	}
	if client.channel != "test" {
		t.Fatalf("expected channel to be set")
	}
	// Optional: Verify new fields are set correctly
	if client.username != "test_user" {
		t.Fatalf("expected username to be set")
	}
	if client.token != "oauth:12345" {
		t.Fatalf("expected token to be set")
	}

	if client.DataChan == nil {
		t.Fatalf("expected DataChan to be initialized")
	}
	if cap(client.DataChan) != 1000 {
		t.Fatalf("expected DataChan buffer 1000, got %d", cap(client.DataChan))
	}
}
