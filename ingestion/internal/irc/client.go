package irc

import (
	"log"
	"net/url"
	"time"

	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/metrics" // Import this!
	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/parser"
	"github.com/gorilla/websocket"
)

type Client struct {
	conn           *websocket.Conn
	url            string
	channel        string
	reconnectDelay time.Duration

	// NEW: The "Conveyor Belt".
	// We send parsed messages out here so 'main.go' can handle them.
	// The '1000' means it can hold 1000 messages before it gets full (Backpressure).
	DataChan chan parser.Message
}

func NewClient(wsUrl, channelName string) *Client {
	return &Client{
		url:            wsUrl,
		channel:        channelName,
		reconnectDelay: 1 * time.Second,
		DataChan:       make(chan parser.Message, 1000), // Initialize the channel
	}
}

func (c *Client) Connect() {
	for {
		log.Printf("Connecting to %s...", c.url)
		u, _ := url.Parse(c.url)
		conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)

		if err != nil {
			log.Printf("Connection failed: %v. Retrying in %v...", err, c.reconnectDelay)
			time.Sleep(c.reconnectDelay)
			c.reconnectDelay *= 2
			if c.reconnectDelay > 30*time.Second {
				c.reconnectDelay = 30 * time.Second
			}
			continue
		}

		c.conn = conn
		c.reconnectDelay = 1 * time.Second
		log.Println("Connected to Twitch IRC!")

		c.authenticate()

		// If readPump returns error, loop back and reconnect
		err = c.readPump()
		if err != nil {
			log.Printf("Connection lost: %v", err)
			c.conn.Close()
		}
	}
}

func (c *Client) authenticate() {
	c.conn.WriteMessage(websocket.TextMessage, []byte("CAP REQ :twitch.tv/tags twitch.tv/commands"))
	c.conn.WriteMessage(websocket.TextMessage, []byte("PASS SCHMOOPIIE"))
	c.conn.WriteMessage(websocket.TextMessage, []byte("NICK justinfan12345"))
	c.conn.WriteMessage(websocket.TextMessage, []byte("USER justinfan12345 8 * :justinfan12345"))
	c.conn.WriteMessage(websocket.TextMessage, []byte("JOIN #"+c.channel))
}

func (c *Client) readPump() error {
	for {
		_, rawBytes, err := c.conn.ReadMessage()
		if err != nil {
			return err
		}

		// 1. Parse the line using our new tool (byte-level)
		msg := parser.ParseBytes(rawBytes)

		// 2. If it returned nil, it was junk (JOIN/PART). Ignore it.
		if msg == nil {
			continue
		}

		// 3. If it is a PING, we MUST Pong back or Twitch disconnects us.
		if msg.Type == "PING" {
			log.Println("Received PING, sending PONG...")
			c.conn.WriteMessage(websocket.TextMessage, []byte("PONG :tmi.twitch.tv"))
			continue
		}

		// METRICS UPDATE:
		// Update Queue Depth Gauge
		metrics.QueueDepth.Set(float64(len(c.DataChan)))
		// 4. If it is CHAT, put it on the Conveyor Belt.
		// This blocks producers when the belt is full (backpressure).
		c.DataChan <- *msg

		// Success! Message added to channel.
		metrics.MessagesProcessed.Inc() // +1 Processed
	}
}
