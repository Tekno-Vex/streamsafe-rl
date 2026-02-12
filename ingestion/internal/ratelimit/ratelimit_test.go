package ratelimit

import (
	"testing"
	"time"
)

func TestNewPerChannelLimiter(t *testing.T) {
	limiter := NewPerChannelLimiter(1000)
	if limiter == nil {
		t.Fatal("expected limiter")
	}
	if len(limiter.limiters) != 0 {
		t.Fatalf("expected empty limiter map, got %d", len(limiter.limiters))
	}
}

func TestAllowCreatesLimiter(t *testing.T) {
	limiter := NewPerChannelLimiter(2)
	if !limiter.Allow("ch1") {
		t.Fatal("expected first call to allow")
	}
	if len(limiter.limiters) != 1 {
		t.Fatalf("expected limiter to be created")
	}
}

func TestPerChannelIsolation(t *testing.T) {
	limiter := NewPerChannelLimiter(2)

	if !limiter.Allow("ch1") || !limiter.Allow("ch1") {
		t.Fatal("expected burst for ch1")
	}
	if limiter.Allow("ch1") {
		t.Fatal("expected ch1 to be rate limited")
	}

	if !limiter.Allow("ch2") || !limiter.Allow("ch2") {
		t.Fatal("expected burst for ch2")
	}
	if limiter.Allow("ch2") {
		t.Fatal("expected ch2 to be rate limited")
	}
}

func TestTokenRefill(t *testing.T) {
	limiter := NewPerChannelLimiter(10) // 1 token per 100ms

	for i := 0; i < 10; i++ {
		if !limiter.Allow("ch1") {
			t.Fatalf("expected burst allow at %d", i)
		}
	}
	if limiter.Allow("ch1") {
		t.Fatal("expected rate limited after burst")
	}

	time.Sleep(120 * time.Millisecond)
	if !limiter.Allow("ch1") {
		t.Fatal("expected token refill")
	}
}
