package search

import (
	"context"
	"net/url"
	"strings"
	"time"

	"modelrun/internal/agent"
)

func callCtx(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, 12*time.Second)
}

// Ground looks a factual question up before the model speaks. The caller
// has already decided this question is a fact. prompt is the turn the model
// should answer. sources is a short list to show under that answer.
// An empty prompt means the call was interrupted or the question was empty.
func (c *Client) Ground(ctx context.Context, question string, status func(string), card func(label, text string)) (prompt, sources string, err error) {
	if strings.TrimSpace(question) == "" {
		return "", "", nil
	}
	var trace []item
	for _, src := range sourcesFor(question) {
		if ctx.Err() != nil {
			return "", "", ctx.Err()
		}
		if status != nil {
			status(src.Label + "…")
		}
		q := agent.SearchQuery(question)
		if agent.Live(question) {
			q = agent.PlanQuery(question)
		}
		call, stop := callCtx(ctx)
		out, err := c.Lookup(call, q, src.ID)
		stop()
		if ctx.Err() != nil {
			return "", "", ctx.Err()
		}
		if err != nil && strings.TrimSpace(out) == "" {
			out = "The lookup failed."
		}
		echo := frameEcho(out, question) || frameEcho(out, q)
		if !echo && sourceKept(out) {
			out = filterHitText(q, out)
		}
		kept := !echo && sourceKept(out)
		text := ""
		if kept {
			text = clipLines(out, 1800)
		}
		trace = append(trace, item{
			kind: "tool", tool: "search", label: src.Label, text: text, kept: kept,
		})
		if card != nil && kept {
			card(src.Label, cardClip(out, 500))
		}
	}
	if err := c.readSources(ctx, &trace, status, card); err != nil {
		return "", "", err
	}
	if ctx.Err() != nil {
		return "", "", ctx.Err()
	}
	prompt, sources = modelBrief(question, trace)
	return prompt, sources, nil
}

func cardClip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	cut := clipBytes(s, n)
	if i := strings.LastIndex(cut, "\n\n"); i > 40 {
		return strings.TrimSpace(cut[:i])
	}
	if i := strings.LastIndex(cut, ". "); i > 80 {
		return strings.TrimSpace(cut[:i+1])
	}
	return strings.TrimSpace(cut)
}

func skipPage(raw string) bool {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return true
	}
	host := strings.TrimPrefix(strings.ToLower(u.Hostname()), "www.")
	switch host {
	case "jstor.org", "google.com", "scholar.google.com", "web.archive.org", "archive.org", "doi.org", "dx.doi.org":
		return true
	}
	return strings.HasSuffix(host, ".google.com") || strings.Contains(host, "duckduckgo.com")
}

// sourcesFor is one engine. News, weather, and prices are a web search.
// A stable fact is Wikipedia. The other engines were adding pages about a
// different subject.
func sourcesFor(question string) []agent.Source {
	if agent.Live(question) {
		return []agent.Source{{ID: "duckduckgo", Label: "DuckDuckGo"}}
	}
	return []agent.Source{{ID: "wiki", Label: "Wikipedia"}}
}

func filterHitText(q, text string) string {
	hits := parseHits(text)
	if len(hits) == 0 {
		return text
	}
	kept := related(q, hits)
	if len(kept) == 0 {
		return ""
	}
	return formatHits(kept)
}

func (c *Client) readSources(ctx context.Context, trace *[]item, status func(string), card func(label, text string)) error {
	for _, it := range *trace {
		if it.page {
			return nil
		}
	}
	var groups [][]hit
	for _, it := range *trace {
		if it.kind != "tool" || it.tool != "search" || !it.kept {
			continue
		}
		var group []hit
		for _, h := range parseHits(it.text) {
			if h.url != "" {
				group = append(group, h)
			}
		}
		if len(group) > 0 {
			groups = append(groups, group)
		}
	}
	var hits []hit
	seen := map[string]bool{}
	for len(hits) < 1 {
		moved := false
		for i := range groups {
			for len(groups[i]) > 0 && seen[groups[i][0].url] {
				groups[i] = groups[i][1:]
			}
			if len(groups[i]) == 0 {
				continue
			}
			h := groups[i][0]
			groups[i] = groups[i][1:]
			moved = true
			if seen[h.url] || skipPage(h.url) {
				continue
			}
			seen[h.url] = true
			hits = append(hits, h)
			break
		}
		if !moved {
			break
		}
	}
	for _, h := range hits {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		label := h.title
		if label == "" {
			label = "Page"
		}
		if status != nil {
			status("Reading " + label + "…")
		}
		call, stop := callCtx(ctx)
		out, err := c.Read(call, h.url)
		stop()
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err != nil && strings.TrimSpace(out) == "" {
			out = "The lookup failed."
		}
		prose := ""
		if sourceKept(out) {
			prose = pageProse(out)
		}
		text := ""
		kept := prose != ""
		if kept {
			text = clipLines(prose, pageChars)
		}
		*trace = append(*trace, item{
			kind: "tool", tool: "gather", label: label, page: true, url: h.url,
			text: text, kept: kept,
		})
	}
	return nil
}
