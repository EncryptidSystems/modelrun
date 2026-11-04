// Package vm is the EncryptidVM client the site uses. Commands run in a
// TunnelTug guest through the MCP endpoint, never on this machine.
package vm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
)

const protocol = "2025-03-26"

// DefaultEndpoint is the console the site talks to.
const DefaultEndpoint = "http://127.0.0.1:8870/mcp"

var exitCode = regexp.MustCompile(`\nexit: (-?\d+)`)

// Client is one MCP session. The first command opens a sandbox on the first host.
type Client struct {
	Endpoint string
	Host     string
	HTTP     *http.Client

	mu      sync.Mutex
	session string
	open    bool
	seq     int
}

// New reads MODELRUN_VM, otherwise the site's endpoint.
func New() *Client {
	ep := os.Getenv("MODELRUN_VM")
	if ep == "" {
		ep = DefaultEndpoint
	}
	return &Client{Endpoint: ep, HTTP: http.DefaultClient}
}

type rpc struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  any             `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (c *Client) post(ctx context.Context, method string, params any, notify bool) (json.RawMessage, error) {
	c.mu.Lock()
	c.seq++
	id := c.seq
	session := c.session
	c.mu.Unlock()
	msg := rpc{JSONRPC: "2.0", Method: method, Params: params}
	if !notify {
		msg.ID = id
	}
	body, err := json.Marshal(msg)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("MCP-Protocol-Version", protocol)
	if session != "" {
		req.Header.Set("Mcp-Session-Id", session)
	}
	ht := c.HTTP
	if ht == nil {
		ht = http.DefaultClient
	}
	res, err := ht.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if sid := res.Header.Get("Mcp-Session-Id"); sid != "" {
		c.mu.Lock()
		c.session = sid
		c.mu.Unlock()
	}
	if notify {
		if res.StatusCode >= 300 {
			return nil, fmt.Errorf("%s", strings.TrimSpace(string(raw)))
		}
		return nil, nil
	}
	if res.StatusCode >= 300 {
		t := strings.TrimSpace(string(raw))
		if t == "" {
			t = res.Status
		}
		if len(t) > 200 {
			t = t[:200]
		}
		return nil, fmt.Errorf("%s", t)
	}
	parsed, err := rpcBody(string(raw))
	if err != nil {
		return nil, err
	}
	if parsed.Error != nil {
		if parsed.Error.Message == "" {
			return nil, fmt.Errorf("error")
		}
		return nil, fmt.Errorf("%s", parsed.Error.Message)
	}
	return parsed.Result, nil
}

func rpcBody(text string) (rpc, error) {
	t := strings.TrimSpace(text)
	var out rpc
	if strings.HasPrefix(t, "{") {
		err := json.Unmarshal([]byte(t), &out)
		return out, err
	}
	var last *rpc
	for _, line := range strings.Split(t, "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var one rpc
		if json.Unmarshal([]byte(strings.TrimSpace(line[5:])), &one) == nil {
			last = &one
		}
	}
	if last == nil {
		return rpc{}, fmt.Errorf("unreadable answer")
	}
	return *last, nil
}

func toolText(raw json.RawMessage) (string, bool, error) {
	var res struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return "", false, err
	}
	var b strings.Builder
	for _, c := range res.Content {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(c.Text)
	}
	return b.String(), res.IsError, nil
}

func (c *Client) tool(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	raw, err := c.post(ctx, "tools/call", map[string]any{"name": name, "arguments": args}, false)
	if err != nil {
		return "", false, err
	}
	return toolText(raw)
}

func (c *Client) ensure(ctx context.Context) error {
	c.mu.Lock()
	have := c.session != ""
	opened := c.open
	c.mu.Unlock()
	if !have {
		if _, err := c.post(ctx, "initialize", map[string]any{
			"protocolVersion": protocol,
			"capabilities":    map[string]any{},
			"clientInfo":      map[string]any{"name": "modelrun", "version": "1"},
		}, false); err != nil {
			return err
		}
		if _, err := c.post(ctx, "notifications/initialized", nil, true); err != nil {
			return err
		}
		c.mu.Lock()
		c.open = false
		opened = false
		c.mu.Unlock()
	}
	if opened {
		return nil
	}
	host := c.Host
	if host == "" {
		text, bad, err := c.tool(ctx, "list_hosts", map[string]any{})
		if err != nil {
			return err
		}
		if bad {
			return fmt.Errorf("%s", text)
		}
		host = strings.TrimSpace(strings.Split(strings.Split(text, "\n")[0], "\t")[0])
		if host == "" || host == "no hosts" {
			return fmt.Errorf("EncryptidVM has no host. Run tunneltug vm up.")
		}
	}
	text, bad, err := c.tool(ctx, "sandbox_open", map[string]any{"host": host})
	if err != nil {
		return err
	}
	if bad {
		return fmt.Errorf("%s", text)
	}
	c.mu.Lock()
	c.open = true
	c.mu.Unlock()
	return nil
}

// Exec runs command in the open sandbox. The result is the guest's text.
func (c *Client) Exec(ctx context.Context, command string) (string, int) {
	cmd := strings.TrimSpace(command)
	if cmd == "" {
		return "Missing command.", 1
	}
	run := func() (string, bool, error) {
		if err := c.ensure(ctx); err != nil {
			return "", false, err
		}
		return c.tool(ctx, "sandbox_exec", map[string]any{"command": cmd})
	}
	text, bad, err := run()
	if err == nil && bad && regexp.MustCompile(`(?i)not open|session`).MatchString(text) {
		c.mu.Lock()
		c.session = ""
		c.open = false
		c.mu.Unlock()
		text, bad, err = run()
	}
	if err != nil {
		c.mu.Lock()
		c.session = ""
		c.open = false
		c.mu.Unlock()
		why := err.Error()
		if regexp.MustCompile(`(?i)Failed to fetch|connection refused|connectex|No connection|context canceled`).MatchString(why) {
			if strings.Contains(strings.ToLower(why), "cancel") {
				return "Interrupted.", 1
			}
			return "EncryptidVM is not answering at " + c.Endpoint + ". Start it with: encryptidvm serve", 1
		}
		return "EncryptidVM: " + why, 1
	}
	m := exitCode.FindStringSubmatch("\n" + text)
	code := 0
	if m != nil {
		fmt.Sscanf(m[1], "%d", &code)
	} else if bad {
		code = 1
	}
	if text == "" {
		text = "(no output)"
	}
	return text, code
}
