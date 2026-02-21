package irc

import (
	"log"
	"net/url"
	"time"

	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/metrics"
	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/parser"
	"github.com/gorilla/websocket"
)

type Client struct {
	conn           *websocket.Conn
	url            string
	channel        string
	username       string
	token          string
	reconnectDelay time.Duration
	DataChan       chan parser.Message
}

func NewClient(wsUrl, channelName, username, token string) *Client {
	return &Client{
		url:            wsUrl,
		channel:        channelName,
		username:       username,
		token:          token,
		reconnectDelay: 1 * time.Second,
		DataChan:       make(chan parser.Message, 1000),
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

		err = c.readPump()
		if err != nil {
			log.Printf("Connection lost: %v", err)
			c.conn.Close()
		}
	}
}

func (c *Client) authenticate() {
	log.Printf("Authenticating as %s...", c.username)
	c.conn.WriteMessage(websocket.TextMessage, []byte("CAP REQ :twitch.tv/tags twitch.tv/commands"))
	c.conn.WriteMessage(websocket.TextMessage, []byte("PASS "+c.token))
	c.conn.WriteMessage(websocket.TextMessage, []byte("NICK "+c.username))
	c.conn.WriteMessage(websocket.TextMessage, []byte("JOIN #"+c.channel))
}

func (c *Client) readPump() error {
	for {
		_, rawBytes, err := c.conn.ReadMessage()
		if err != nil {
			return err
		}

		// CRITICAL: Log everything Twitch sends us
		log.Printf("[DEBUG] RAW: %s", string(rawBytes))

		// Parse the line
		msg := parser.ParseBytes(rawBytes)

		if msg == nil {
			continue
		}

		if msg.Type == "PING" {
			log.Println("Received PING, sending PONG...")
			c.conn.WriteMessage(websocket.TextMessage, []byte("PONG :tmi.twitch.tv"))
			continue
		}

		if msg.Type == "JOIN" {
			log.Printf("Successfully JOINED channel: %s", msg.Channel)
			continue
		}

		if msg.Type == "NOTICE" {
			log.Printf("Twitch NOTICE: %s", string(rawBytes))
		}

		metrics.QueueDepth.Set(float64(len(c.DataChan)))
		c.DataChan <- *msg
		metrics.MessagesProcessed.Inc()
	}
}
