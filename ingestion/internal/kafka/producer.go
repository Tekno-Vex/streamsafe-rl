package kafka

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/Tekno-Vex/streamsafe-rl/ingestion/internal/parser"
	"github.com/segmentio/kafka-go"
)

type Producer struct {
	writer *kafka.Writer
}

// NewProducer creates a Kafka producer
func NewProducer(brokerAddr, topic string) *Producer {
	return &Producer{
		writer: &kafka.Writer{
			Addr:         kafka.TCP(brokerAddr),
			Topic:        topic,
			Balancer:     &kafka.LeastBytes{},
			BatchTimeout: 10 * time.Millisecond,
			BatchSize:    100,
		},
	}
}

// PublishMessage sends a message to Kafka
func (p *Producer) PublishMessage(msg parser.Message) error {
	// Convert message to JSON
	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("Failed to marshal message: %v", err)
		return err
	}

	// Send to Kafka
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = p.writer.WriteMessages(ctx, kafka.Message{
		Key:   []byte(msg.User),
		Value: data,
	})

	if err != nil {
		log.Printf("Failed to publish to Kafka: %v", err)
		return err
	}

	return nil
}

// Close shuts down the producer
func (p *Producer) Close() error {
	return p.writer.Close()
}
