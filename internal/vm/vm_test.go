package vm

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExecOpensSandbox(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s := string(b)
		switch {
		case strings.Contains(s, "notifications/initialized"):
			seen = append(seen, "initialized")
			w.WriteHeader(202)
		case strings.Contains(s, "initialize"):
			seen = append(seen, "initialize")
			w.Header().Set("Mcp-Session-Id", "s1")
			w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{}}`))
		case strings.Contains(s, "list_hosts"):
			seen = append(seen, "list_hosts")
			w.Write([]byte(`{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"eda\t1"}]}}`))
		case strings.Contains(s, "sandbox_open"):
			seen = append(seen, "sandbox_open")
			w.Write([]byte(`{"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"sandbox open"}]}}`))
		case strings.Contains(s, "sandbox_exec"):
			seen = append(seen, "sandbox_exec")
			if !strings.Contains(s, "echo hi") {
				t.Errorf("command %s", s)
			}
			w.Write([]byte(`{"jsonrpc":"2.0","id":5,"result":{"content":[{"type":"text","text":"hi\nexit: 0"}]}}`))
		default:
			t.Errorf("unexpected %s", s)
			w.WriteHeader(400)
		}
	}))
	defer srv.Close()
	c := &Client{Endpoint: srv.URL, HTTP: srv.Client()}
	out, code := c.Exec(t.Context(), "echo hi")
	if code != 0 || !strings.Contains(out, "hi") {
		t.Fatalf("code %d out %q seen %v", code, out, seen)
	}
	want := []string{"initialize", "initialized", "list_hosts", "sandbox_open", "sandbox_exec"}
	if strings.Join(seen, ",") != strings.Join(want, ",") {
		t.Fatalf("seen %v", seen)
	}
}
