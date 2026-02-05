package parser

import (
	"strings"
	"time"
)

// 1. This is the clean "Form" we want to fill out for every message.
type Message struct {
	Type      string    // "CHAT", "PING", or "SYSTEM"
	User      string    // The username (e.g., "helloimjoe")
	Channel   string    // The channel (e.g., "jasontheween")
	Content   string    // The actual text (e.g., "hiiii")
	Timestamp time.Time // When we received it
}

// 2. This function takes the raw string and fills out the Form.
func Parse(rawLine string) *Message {
	// If the line is empty, ignore it.
	if len(rawLine) == 0 {
		return nil
	}

	// 3. Handle PING messages (Twitch sends these to check if we are alive).
	if strings.HasPrefix(rawLine, "PING") {
		return &Message{Type: "PING", Content: rawLine}
	}

	// 4. If it's not a PING, we split the string by spaces.
	// Raw Format Example:
	// :shwipp_!shwipp_@... PRIVMSG #jasontheween :hiiii
	parts := strings.Split(rawLine, " ")

	if len(parts) < 4 {
		return nil // Not a valid message
	}

	// 5. Look for the "PRIVMSG" command (which means "Private Message", aka Chat).
	// Sometimes the command is in index 2 (if tags are present) or index 1.
	commandIndex := -1
	for i, part := range parts {
		if part == "PRIVMSG" {
			commandIndex = i
			break
		}
	}

	// If we didn't find PRIVMSG, it's a JOIN/PART/SYSTEM message. We ignore those (Filtering).
	if commandIndex == -1 {
		return nil
	}

	// 6. Extract the Username
	// The user part looks like ":shwipp_!..."
	// We take the part before the "!", and remove the ":" at the start.
	userPart := parts[commandIndex-1]
	user := ""
	if strings.Contains(userPart, "!") {
		user = strings.Split(userPart, "!")[0]
		user = strings.TrimPrefix(user, ":") // Remove the leading colon
	}

	// 7. Extract the Channel
	channel := parts[commandIndex+1]
	channel = strings.TrimPrefix(channel, "#")

	// 8. Extract the Content (The chat message)
	// Everything after "PRIVMSG #channel :" is the message.
	// We re-join the rest of the string parts.
	contentStartIndex := commandIndex + 2
	content := strings.Join(parts[contentStartIndex:], " ")
	content = strings.TrimPrefix(content, ":") // Remove the leading colon

	return &Message{
		Type:      "CHAT",
		User:      user,
		Channel:   channel,
		Content:   content,
		Timestamp: time.Now(),
	}
}
