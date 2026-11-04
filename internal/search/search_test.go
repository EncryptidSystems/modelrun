package search

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSearchAndGather(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/ext/search":
			if r.URL.Query().Get("q") != "lake champlain" {
				t.Errorf("q %q", r.URL.Query().Get("q"))
			}
			_, _ = w.Write([]byte(`{"ok":true,"output":"1. Lake Champlain\nhttps://en.wikipedia.org/wiki/Lake_Champlain\nA lake."}`))
		case "/api/ext/gather":
			if !strings.Contains(r.URL.Query().Get("url"), "Lake_Champlain") {
				t.Errorf("url %q", r.URL.Query().Get("url"))
			}
			_, _ = w.Write([]byte(`{"ok":true,"output":"Lake Champlain is a lake."}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	c := &Client{Base: srv.URL, HTTP: srv.Client()}
	got, err := c.Search(t.Context(), "lake champlain", "")
	if err != nil || !strings.Contains(got, "https://en.wikipedia.org/wiki/Lake_Champlain") {
		t.Fatalf("search %q %v", got, err)
	}
	page, err := c.Gather(t.Context(), "https://en.wikipedia.org/wiki/Lake_Champlain")
	if err != nil || page != "Lake Champlain is a lake." {
		t.Fatalf("gather %q %v", page, err)
	}
	if URLs(got)[0] != "https://en.wikipedia.org/wiki/Lake_Champlain" {
		t.Fatalf("urls %v", URLs(got))
	}
}
