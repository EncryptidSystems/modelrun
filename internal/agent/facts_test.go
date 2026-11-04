package agent

import "testing"

func TestNeedsFacts(t *testing.T) {
	yes := []string{
		"what is the capital of France",
		"how to make fried chicken",
		"tell me about the history of the printing press",
		"describe photosynthesis",
		"fried chicken recipe",
	}
	no := []string{
		"hello",
		"what is 2 + 2",
		"who are you",
		"write a poem about a dragon",
		"Help me make fried chicken",
		"describe how a lock works",
	}
	for _, s := range yes {
		if !NeedsFacts(s) {
			t.Errorf("want facts %q", s)
		}
	}
	for _, s := range no {
		if NeedsFacts(s) {
			t.Errorf("want model %q", s)
		}
	}
}

func TestLive(t *testing.T) {
	if !Live("what's the latest news") || !Live("weather in Chicago") || !Live("price of gold") {
		t.Fatal("live")
	}
	if Live("what is the capital of France") || Live("who is the president of France") {
		t.Fatal("encyclopedia")
	}
}

func TestSearchQuery(t *testing.T) {
	if got := SearchQuery("tell me about the history of the printing press"); got != "the history of the printing press" {
		t.Fatalf("press %q", got)
	}
	if got := SearchQuery("what is the capital of France"); got != "the capital of France" {
		t.Fatalf("france %q", got)
	}
	if got := SearchQuery("how to make fried chicken"); got != "make fried chicken" {
		t.Fatalf("chicken %q", got)
	}
}
