// Package search calls the site lookup. The pages are /api/ext/search and
// /api/ext/gather on ai.encryptid.systems, the same handlers the desk uses.
package search

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

// DefaultEndpoint is the public desk. MODELRUN_SEARCH overrides it.
const DefaultEndpoint = "https://ai.encryptid.systems"

// Endpoint is the desk origin, without a trailing slash.
func Endpoint() string {
	if s := strings.TrimSpace(os.Getenv("MODELRUN_SEARCH")); s != "" {
		return strings.TrimRight(s, "/")
	}
	return DefaultEndpoint
}

// Client talks to one desk.
type Client struct {
	Base string
	HTTP *http.Client
}

// New uses Endpoint.
func New() *Client {
	return &Client{Base: Endpoint()}
}

func (c *Client) base() string {
	if c != nil && c.Base != "" {
		return strings.TrimRight(c.Base, "/")
	}
	return Endpoint()
}

func (c *Client) http() *http.Client {
	if c != nil && c.HTTP != nil {
		return c.HTTP
	}
	return &http.Client{Timeout: 30 * time.Second}
}

// Search runs one lookup. source is wiki, duckduckgo, or empty for the desk default.
func (c *Client) Search(ctx context.Context, q, source string) (string, error) {
	q = strings.TrimSpace(q)
	if q == "" {
		return "Missing search query.", nil
	}
	if len(q) > 200 {
		q = q[:200]
	}
	u, err := url.Parse(c.base() + "/api/ext/search")
	if err != nil {
		return "The lookup failed.", err
	}
	qv := u.Query()
	qv.Set("q", q)
	if s := strings.TrimSpace(source); s != "" {
		qv.Set("source", s)
	}
	u.RawQuery = qv.Encode()
	return c.get(ctx, u.String())
}

// Gather reads one public page through the desk.
func (c *Client) Gather(ctx context.Context, raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "Missing address.", nil
	}
	u, err := url.Parse(c.base() + "/api/ext/gather")
	if err != nil {
		return "The lookup failed.", err
	}
	qv := u.Query()
	qv.Set("url", raw)
	u.RawQuery = qv.Encode()
	return c.get(ctx, u.String())
}

func (c *Client) get(ctx context.Context, raw string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, raw, nil)
	if err != nil {
		return "The lookup failed.", err
	}
	req.Header.Set("Accept", "application/json")
	res, err := c.http().Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "Interrupted.", ctx.Err()
		}
		return "The lookup failed.", err
	}
	defer res.Body.Close()
	var body struct {
		OK     bool   `json:"ok"`
		Output string `json:"output"`
	}
	if json.NewDecoder(io.LimitReader(res.Body, 8000)).Decode(&body) != nil {
		return "The lookup failed.", nil
	}
	if body.Output == "" && !body.OK {
		return "The lookup failed.", nil
	}
	return body.Output, nil
}

var urlRE = regexp.MustCompile("(?i)https?://[^\\s<>\"'`" + "`" + ")\\]]+")

// URLs returns the web addresses in text, the same set gather is allowed to read.
func URLs(text string) []string {
	var out []string
	seen := map[string]bool{}
	for _, m := range urlRE.FindAllString(text, -1) {
		if u := Canonical(m); u != "" && !seen[u] {
			seen[u] = true
			out = append(out, u)
		}
	}
	return out
}

// Canonical is one address in the form the page compares.
func Canonical(raw string) string {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimRight(raw, ".,;:!?")
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return ""
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return ""
	}
	return u.String()
}
