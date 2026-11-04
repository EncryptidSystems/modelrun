// Package api is the Ollama-shaped HTTP surface.
//
//	GET  /api/tags
//	GET  /api/ps
//	GET  /api/version
//	POST /api/show
//	POST /api/chat
//	POST /api/generate
package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"modelrun/internal/catalog"
	"modelrun/internal/generate"
	"modelrun/internal/mark"
)

const Version = "modelrun 0.2"

// Handler serves the API.
func Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/version", version)
	mux.HandleFunc("GET /api/tags", tags)
	mux.HandleFunc("GET /api/ps", ps)
	mux.HandleFunc("POST /api/show", show)
	mux.HandleFunc("POST /api/chat", chat)
	mux.HandleFunc("POST /api/generate", gen)
	return mux
}

func version(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, map[string]string{"version": Version})
}

func tags(w http.ResponseWriter, _ *http.Request) {
	all, err := catalog.List()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	type row struct {
		Name    string         `json:"name"`
		Model   string         `json:"model"`
		Size    int64          `json:"size"`
		Details map[string]any `json:"details"`
	}
	var models []row
	for _, m := range all {
		models = append(models, row{
			Name:  m.Name,
			Model: m.Name,
			Size:  m.Bytes,
			Details: map[string]any{
				"family":             m.Kind,
				"parameter_size":     m.Detail,
				"quantization_level": m.Family,
				"parent_model":       "",
			},
		})
	}
	writeJSON(w, map[string]any{"models": models})
}

func ps(w http.ResponseWriter, _ *http.Request) {
	type row struct {
		Name    string    `json:"name"`
		Model   string    `json:"model"`
		Expires time.Time `json:"expires_at"`
	}
	var models []row
	for name, at := range generate.Loaded() {
		models = append(models, row{Name: name, Model: name, Expires: at.Add(24 * time.Hour)})
	}
	writeJSON(w, map[string]any{"models": models})
}

func show(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	name := req.Name
	if name == "" {
		name = req.Model
	}
	m, ok, err := catalog.Get(name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !ok {
		http.Error(w, "no model "+name, http.StatusNotFound)
		return
	}
	writeJSON(w, map[string]any{
		"modelfile": m.Path,
		"details":   m,
	})
}

type options struct {
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`
	NumPredict  *int     `json:"num_predict"`
}

func optFrom(o options) generate.Opt {
	opt := generate.Opt{ConfigSample: true}
	if o.NumPredict != nil && *o.NumPredict > 0 {
		opt.N = *o.NumPredict
	}
	if o.Temperature != nil {
		opt.Temp = *o.Temperature
		opt.TempSet = true
	}
	if o.TopP != nil {
		opt.TopP = *o.TopP
		opt.TopSet = true
	}
	return opt
}

func chat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		Stream  *bool   `json:"stream"`
		Options options `json:"options"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	turns := make([]generate.Turn, 0, len(req.Messages))
	for _, m := range req.Messages {
		turns = append(turns, generate.Turn{Role: m.Role, Content: m.Content})
	}
	opt := optFrom(req.Options)
	stream := req.Stream != nil && *req.Stream
	if !stream {
		reply, err := generate.Chat(req.Model, turns, opt, nil)
		if err != nil {
			writeGenErr(w, err)
			return
		}
		body := map[string]any{
			"model":      reply.Model,
			"created_at": time.Now().UTC().Format(time.RFC3339),
			"message": map[string]string{
				"role":    "assistant",
				"content": reply.Text,
			},
			"done": true,
		}
		if s, ok := watermark(req.Model, reply.Text); ok {
			body["watermark"] = s
		}
		writeJSON(w, body)
		return
	}
	writeNDJSON(w, req.Model, true, func(emit func(string)) (string, error) {
		reply, err := generate.Chat(req.Model, turns, opt, emit)
		if err != nil {
			return "", err
		}
		return reply.Model, nil
	})
}

func gen(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model   string  `json:"model"`
		Prompt  string  `json:"prompt"`
		Stream  *bool   `json:"stream"`
		Options options `json:"options"`
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	opt := optFrom(req.Options)
	stream := req.Stream != nil && *req.Stream
	if !stream {
		reply, err := generate.Complete(req.Model, req.Prompt, opt, nil)
		if err != nil {
			writeGenErr(w, err)
			return
		}
		body := map[string]any{
			"model":      reply.Model,
			"created_at": time.Now().UTC().Format(time.RFC3339),
			"response":   reply.Text,
			"done":       true,
		}
		if s, ok := watermark(req.Model, reply.Text); ok {
			body["watermark"] = s
		}
		writeJSON(w, body)
		return
	}
	writeNDJSON(w, req.Model, false, func(emit func(string)) (string, error) {
		reply, err := generate.Complete(req.Model, req.Prompt, opt, emit)
		if err != nil {
			return "", err
		}
		return reply.Model, nil
	})
}

// writeNDJSON sends one object per piece, then a done object.
// chat shapes the piece as message.content. A completion uses response.
func writeNDJSON(w http.ResponseWriter, model string, chat bool, run func(func(string)) (string, error)) {
	fl, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "application/x-ndjson")
	enc := json.NewEncoder(w)
	tok, cfg := filesOf(model)
	var chk *mark.Checker
	if tok != "" {
		chk, _ = mark.Cached(tok, cfg)
	}
	var acc strings.Builder
	var cur mark.Score
	started := false
	name, err := run(func(piece string) {
		started = true
		acc.WriteString(piece)
		if chk != nil {
			cur = chk.Score(acc.String())
		}
		obj := map[string]any{
			"model":      model,
			"created_at": time.Now().UTC().Format(time.RFC3339),
			"done":       false,
		}
		if chat {
			obj["message"] = map[string]string{"role": "assistant", "content": piece}
		} else {
			obj["response"] = piece
		}
		if cur.Words > 0 {
			obj["watermark"] = cur
		}
		_ = enc.Encode(obj)
		if fl != nil {
			fl.Flush()
		}
	})
	if err != nil && !started {
		writeGenErr(w, err)
		return
	}
	if name != "" {
		model = name
	}
	obj := map[string]any{
		"model":      model,
		"created_at": time.Now().UTC().Format(time.RFC3339),
		"done":       true,
	}
	if chat {
		obj["message"] = map[string]string{"role": "assistant", "content": ""}
	} else {
		obj["response"] = ""
	}
	if err != nil {
		obj["error"] = err.Error()
	}
	if cur.Words > 0 {
		obj["watermark"] = cur
	}
	_ = enc.Encode(obj)
	if fl != nil {
		fl.Flush()
	}
}

func writeGenErr(w http.ResponseWriter, err error) {
	code := http.StatusBadGateway
	if errors.Is(err, mark.ErrMarked) {
		code = http.StatusBadRequest
	}
	http.Error(w, err.Error(), code)
}

func filesOf(name string) (string, string) {
	if m, err := catalog.Load(name); err == nil {
		return m.Tokenizer, m.TokenizerConfig
	}
	if m, ok, err := catalog.Get(name); err == nil && ok {
		return m.Tokenizer, m.TokenizerConfig
	}
	return "", ""
}

func watermark(name, text string) (mark.Score, bool) {
	tok, cfg := filesOf(name)
	c, err := mark.Cached(tok, cfg)
	if err != nil || c == nil {
		return mark.Score{}, false
	}
	s := c.Score(text)
	if s.Words == 0 {
		return mark.Score{}, false
	}
	return s, true
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}
