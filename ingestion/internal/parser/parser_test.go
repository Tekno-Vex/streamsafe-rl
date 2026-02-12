package parser

import "testing"

func TestParsePING(t *testing.T) {
	msg := ParseBytes([]byte("PING :tmi.twitch.tv"))
	if msg == nil {
		t.Fatal("expected PING message")
	}
	if msg.Type != "PING" {
		t.Fatalf("expected Type PING, got %q", msg.Type)
	}
}

func TestParsePRIVMSG(t *testing.T) {
	raw := []byte(":user!user@user.tmi.twitch.tv PRIVMSG #channel :hello world")
	msg := ParseBytes(raw)
	if msg == nil {
		t.Fatal("expected CHAT message")
	}
	if msg.Type != "CHAT" {
		t.Fatalf("expected Type CHAT, got %q", msg.Type)
	}
	if msg.User != "user" {
		t.Fatalf("expected user 'user', got %q", msg.User)
	}
	if msg.Channel != "channel" {
		t.Fatalf("expected channel 'channel', got %q", msg.Channel)
	}
	if msg.Content != "hello world" {
		t.Fatalf("expected content 'hello world', got %q", msg.Content)
	}
	if string(msg.MessageBytes) != "hello world" {
		t.Fatalf("expected message bytes 'hello world', got %q", string(msg.MessageBytes))
	}
}

func TestParseNonChatIgnored(t *testing.T) {
	raw := []byte(":user!user@user.tmi.twitch.tv JOIN #channel")
	msg := ParseBytes(raw)
	if msg != nil {
		t.Fatalf("expected JOIN to be ignored")
	}
}

func TestParseShortLine(t *testing.T) {
	msg := ParseBytes([]byte("foo bar"))
	if msg != nil {
		t.Fatalf("expected short line to be ignored")
	}
}

func TestParseTagsFormat(t *testing.T) {
	raw := []byte("@badge-info=subscriber/48;user-id=123;room-id=456 :user!user@user.tmi.twitch.tv PRIVMSG #channel :hi")
	msg := ParseBytes(raw)
	if msg == nil {
		t.Fatal("expected CHAT message with tags")
	}
	if msg.Content != "hi" {
		t.Fatalf("expected content 'hi', got %q", msg.Content)
	}
	if msg.UserID != "123" {
		t.Fatalf("expected user-id '123', got %q", msg.UserID)
	}
	if msg.RoomID != "456" {
		t.Fatalf("expected room-id '456', got %q", msg.RoomID)
	}
}
