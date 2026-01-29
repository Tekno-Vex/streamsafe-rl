# Minimal Go Dockerfile
FROM golang:1.21-alpine
WORKDIR /app
COPY . .
# We build the binary from the cmd/ingest directory
RUN go build -o ingestor ./cmd/ingest
CMD ["./ingestor"]