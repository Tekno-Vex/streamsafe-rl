package ratelimit

import (
	"sync"

	"golang.org/x/time/rate"
)

// PerChannelLimiter manages rate limiters per channel.
type PerChannelLimiter struct {
	limiters map[string]*rate.Limiter
	mu       sync.RWMutex
	rps      float64 // messages per second per channel
}

// NewPerChannelLimiter creates a new per-channel rate limiter.
func NewPerChannelLimiter(rps float64) *PerChannelLimiter {
	return &PerChannelLimiter{
		limiters: make(map[string]*rate.Limiter),
		rps:      rps,
	}
}

// Allow checks if a message from a channel is allowed.
func (pcl *PerChannelLimiter) Allow(channelID string) bool {
	pcl.mu.RLock()
	limiter, exists := pcl.limiters[channelID]
	pcl.mu.RUnlock()

	if !exists {
		pcl.mu.Lock()
		limiter, exists = pcl.limiters[channelID]
		if !exists {
			limiter = rate.NewLimiter(rate.Limit(pcl.rps), int(pcl.rps))
			pcl.limiters[channelID] = limiter
		}
		pcl.mu.Unlock()
	}

	return limiter.Allow()
}
