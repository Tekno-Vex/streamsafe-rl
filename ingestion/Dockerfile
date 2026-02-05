# Build Stage
FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY . .
# Build the binary named "ingestor" from the main file
RUN go build -o ingestor ./cmd/ingest

# Run Stage (Small, secure image)
FROM alpine:latest
WORKDIR /root/
COPY --from=builder /app/ingestor .
# Install certificates so we can talk to HTTPS/WSS (Crucial for Twitch)
RUN apk --no-cache add ca-certificates

# Expose the health port
EXPOSE 8080

CMD ["./ingestor"]