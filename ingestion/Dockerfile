# Upgrade to match your local version
FROM golang:1.25-alpine
WORKDIR /app
COPY . .
RUN go build -o ingestor ./cmd/ingest
CMD ["./ingestor"]