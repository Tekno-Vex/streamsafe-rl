package parser

import (
	"bytes"
	"time"
)

// 1. This is the clean "Form" we want to fill out for every message.
type Message struct {
	Type         string    // "CHAT", "PING", or "SYSTEM"
	User         string    // The username (e.g., "helloimjoe")
	UserID       string    // Twitch user-id tag, if present
	RoomID       string    // Twitch room-id tag, if present
	Channel      string    // The channel (e.g., "jasontheween")
	Content      string    // The actual text (e.g., "hiiii")
	MessageBytes []byte    // Raw message bytes for content
	Timestamp    time.Time // When we received it
}

// 2. This function takes the raw string and fills out the Form.
func Parse(rawLine string) *Message {
	return ParseBytes([]byte(rawLine))
}

// ParseBytes parses raw IRC frames without decoding.
func ParseBytes(rawBytes []byte) *Message {
	if len(rawBytes) == 0 {
		return nil
	}

	if bytes.HasPrefix(rawBytes, []byte("PING")) {
		return &Message{Type: "PING", Content: string(rawBytes), Timestamp: time.Now()}
	}

	parts := bytes.Split(rawBytes, []byte(" "))
	if len(parts) < 4 {
		return nil
	}

	commandIndex := -1
	for i, part := range parts {
		if bytes.Equal(part, []byte("PRIVMSG")) {
			commandIndex = i
			break
		}
	}
	if commandIndex == -1 {
		return nil
	}

	var userID, roomID string
	if len(parts) > 0 && len(parts[0]) > 0 && parts[0][0] == '@' {
		userID, roomID = parseTags(parts[0])
	}

	userPart := parts[commandIndex-1]
	user := ""
	if idx := bytes.IndexByte(userPart, '!'); idx != -1 {
		userBytes := userPart[:idx]
		if len(userBytes) > 0 && userBytes[0] == ':' {
			userBytes = userBytes[1:]
		}
		user = string(userBytes)
	}

	channelPart := parts[commandIndex+1]
	if len(channelPart) > 0 && channelPart[0] == '#' {
		channelPart = channelPart[1:]
	}
	channel := string(channelPart)

	contentStartIndex := commandIndex + 2
	contentBytes := []byte{}
	if contentStartIndex < len(parts) {
		contentBytes = bytes.Join(parts[contentStartIndex:], []byte(" "))
		if len(contentBytes) > 0 && contentBytes[0] == ':' {
			contentBytes = contentBytes[1:]
		}
	}

	return &Message{
		Type:         "CHAT",
		User:         user,
		UserID:       userID,
		RoomID:       roomID,
		Channel:      channel,
		Content:      string(contentBytes),
		MessageBytes: contentBytes,
		Timestamp:    time.Now(),
	}
}

func parseTags(tagBytes []byte) (string, string) {
	if len(tagBytes) == 0 || tagBytes[0] != '@' {
		return "", ""
	}

	userID := ""
	roomID := ""
	trimmed := tagBytes[1:]
	items := bytes.Split(trimmed, []byte(";"))
	for _, item := range items {
		if len(item) == 0 {
			continue
		}
		if eqIdx := bytes.IndexByte(item, '='); eqIdx != -1 {
			key := item[:eqIdx]
			value := item[eqIdx+1:]
			if bytes.Equal(key, []byte("user-id")) {
				userID = string(value)
			}
			if bytes.Equal(key, []byte("room-id")) {
				roomID = string(value)
			}
		}
	}

	return userID, roomID
}
