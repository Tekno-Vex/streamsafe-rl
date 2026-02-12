package irc

import "testing"

func TestNewClientDefaults(t *testing.T) {
	client := NewClient("wss://example.com", "test")
	if client == nil {
		t.Fatal("expected client")
	}
	if client.url != "wss://example.com" {
		t.Fatalf("expected url to be set")
	}
	if client.channel != "test" {
		t.Fatalf("expected channel to be set")
	}
	if client.DataChan == nil {
		t.Fatalf("expected DataChan to be initialized")
	}
	if cap(client.DataChan) != 1000 {
		t.Fatalf("expected DataChan buffer 1000, got %d", cap(client.DataChan))
	}
}
