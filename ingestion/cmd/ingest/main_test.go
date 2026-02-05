package main

import "testing"

func TestSanity(t *testing.T) {
	// This is a "Smoke Test" just to verify the CI pipeline is catching tests.
	expected := 1
	actual := 1

	if expected != actual {
		t.Errorf("Math is broken. Expected %d, got %d", expected, actual)
	}
}
