package search

import (
	"context"
	"encoding/json"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

const wikiAPI = "https://en.wikipedia.org/w/api.php"

var (
	tagRE    = regexp.MustCompile(`<[^>]+>`)
	hexEnt   = regexp.MustCompile(`(?i)&#x([0-9a-f]+);`)
	decEnt   = regexp.MustCompile(`&#(\d+);`)
	chromeRE = regexp.MustCompile(`(?i)\b(?:outdated browser|upgrade your browser|upgrade your web browser|enable javascript|javascript is (?:required|disabled)|turn on javascript)\b`)
	notAPage = regexp.MustCompile(`(?i)(?:wikipedia\.org|wikimedia\.org|wikidata\.org|duckduckgo\.com|web\.archive\.org|archive\.org|doi\.org|dx\.doi\.org|books\.google\.|google\.com/books|d-nb\.info|id\.loc\.gov)`)
)

func clean(s string) string {
	s = tagRE.ReplaceAllString(s, " ")
	s = decodeEntities(s)
	return strings.Join(strings.Fields(s), " ")
}

func decodeEntities(s string) string {
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&AMP;", "&")
	s = hexEnt.ReplaceAllStringFunc(s, func(m string) string {
		sub := hexEnt.FindStringSubmatch(m)
		if len(sub) != 2 {
			return m
		}
		n, err := strconv.ParseInt(sub[1], 16, 32)
		return fromCode(n, err)
	})
	s = decEnt.ReplaceAllStringFunc(s, func(m string) string {
		sub := decEnt.FindStringSubmatch(m)
		if len(sub) != 2 {
			return m
		}
		n, err := strconv.ParseInt(sub[1], 10, 32)
		return fromCode(n, err)
	})
	r := strings.NewReplacer(
		"&lt;", "<", "&gt;", ">", "&quot;", `"`, "&apos;", "'",
		"&nbsp;", " ",
		"&rsquo;", "'", "&lsquo;", "'",
		"&rdquo;", `"`, "&ldquo;", `"`,
		"&ndash;", "-", "&mdash;", "-",
		"&LT;", "<", "&GT;", ">", "&QUOT;", `"`, "&APOS;", "'",
		"&NBSP;", " ",
	)
	return html.UnescapeString(r.Replace(s))
}

func fromCode(n int64, err error) string {
	if err != nil || n <= 0 || n > 0x10ffff {
		return ""
	}
	return string(rune(n))
}

type hit struct {
	title   string
	url     string
	snippet string
}

func formatHits(hits []hit) string {
	var list []hit
	for _, h := range hits {
		if h.title == "" || h.url == "" {
			continue
		}
		list = append(list, h)
		if len(list) == 5 {
			break
		}
	}
	if len(list) == 0 {
		return ""
	}
	var b strings.Builder
	for i, h := range list {
		if i > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(strconv.Itoa(i + 1))
		b.WriteString(". ")
		b.WriteString(h.title)
		b.WriteByte('\n')
		b.WriteString(h.url)
		if h.snippet != "" {
			b.WriteByte('\n')
			b.WriteString(h.snippet)
		}
	}
	s := b.String()
	if len(s) > 1800 {
		s = s[:1800]
		for !utf8.ValidString(s) && len(s) > 0 {
			s = s[:len(s)-1]
		}
	}
	return s
}

func parseWikiSearch(raw []byte) []hit {
	var data struct {
		Query struct {
			Search []struct {
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
			} `json:"search"`
		} `json:"query"`
	}
	if json.Unmarshal(raw, &data) != nil {
		return nil
	}
	var hits []hit
	for _, row := range data.Query.Search {
		title := clean(row.Title)
		if title == "" {
			continue
		}
		if len(title) > 160 {
			title = clipBytes(title, 160)
		}
		snip := clean(row.Snippet)
		if len(snip) > 180 {
			snip = clipBytes(snip, 180)
		}
		hits = append(hits, hit{
			title:   title,
			url:     "https://en.wikipedia.org/wiki/" + url.PathEscape(strings.ReplaceAll(title, " ", "_")),
			snippet: snip,
		})
		if len(hits) >= 5 {
			break
		}
	}
	return hits
}

func parseWikiExtract(raw []byte) string {
	var data struct {
		Query struct {
			Pages map[string]struct {
				Extract string `json:"extract"`
				Missing any    `json:"missing"`
			} `json:"pages"`
		} `json:"query"`
	}
	if json.Unmarshal(raw, &data) != nil || len(data.Query.Pages) == 0 {
		return ""
	}
	var extract string
	var missing bool
	for _, page := range data.Query.Pages {
		extract = page.Extract
		missing = page.Missing != nil
		break
	}
	if extract == "" || missing {
		return ""
	}
	var paras []string
	for _, para := range strings.Split(extract, "\n") {
		t := clean(para)
		if t == "" || chromeRE.MatchString(t) {
			continue
		}
		if len(strings.Fields(t)) < 6 {
			continue
		}
		paras = append(paras, t)
	}
	body := strings.Join(paras, "\n\n")
	if len(body) > 6000 {
		body = clipBytes(body, 6000)
	}
	return body
}

func parseWikiExtlinks(raw []byte) []hit {
	var data struct {
		Query struct {
			Pages map[string]struct {
				Extlinks []map[string]string `json:"extlinks"`
			} `json:"pages"`
		} `json:"query"`
	}
	if json.Unmarshal(raw, &data) != nil {
		return nil
	}
	var rows []map[string]string
	for _, page := range data.Query.Pages {
		rows = page.Extlinks
		break
	}
	var hits []hit
	hosts := map[string]bool{}
	for _, row := range rows {
		u := row["*"]
		if u == "" {
			u = row["url"]
		}
		if !strings.HasPrefix(strings.ToLower(u), "https://") || notAPage.MatchString(u) {
			continue
		}
		parsed, err := url.Parse(u)
		if err != nil {
			continue
		}
		host := strings.TrimPrefix(strings.ToLower(parsed.Hostname()), "www.")
		if host == "" || hosts[host] {
			continue
		}
		hosts[host] = true
		hits = append(hits, hit{title: host, url: u})
		if len(hits) >= 3 {
			break
		}
	}
	return hits
}

func parseDdgInstant(raw []byte) []hit {
	var data struct {
		Heading       string            `json:"Heading"`
		AbstractURL   string            `json:"AbstractURL"`
		AbstractText  string            `json:"AbstractText"`
		RelatedTopics []json.RawMessage `json:"RelatedTopics"`
	}
	if json.Unmarshal(raw, &data) != nil {
		return nil
	}
	var hits []hit
	seen := map[string]bool{}
	add := func(title, rawURL, snippet string) {
		name := clean(title)
		if len(name) > 160 {
			name = clipBytes(name, 160)
		}
		u := strings.TrimSpace(rawURL)
		if name == "" || seen[u] || notAPage.MatchString(u) {
			return
		}
		if !strings.HasPrefix(strings.ToLower(u), "http://") && !strings.HasPrefix(strings.ToLower(u), "https://") {
			return
		}
		seen[u] = true
		snip := clean(snippet)
		if len(snip) > 180 {
			snip = clipBytes(snip, 180)
		}
		hits = append(hits, hit{title: name, url: u, snippet: snip})
	}
	if data.AbstractURL != "" {
		title := data.Heading
		if title == "" {
			title = data.AbstractURL
		}
		add(title, data.AbstractURL, data.AbstractText)
	}
	var walk func([]json.RawMessage)
	walk = func(list []json.RawMessage) {
		for _, raw := range list {
			var item struct {
				Text     string            `json:"Text"`
				FirstURL string            `json:"FirstURL"`
				Topics   []json.RawMessage `json:"Topics"`
			}
			if json.Unmarshal(raw, &item) != nil {
				continue
			}
			if len(item.Topics) > 0 {
				walk(item.Topics)
				continue
			}
			add(item.Text, item.FirstURL, "")
		}
	}
	walk(data.RelatedTopics)
	if len(hits) > 5 {
		hits = hits[:5]
	}
	return hits
}

func wikiTitle(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return ""
	}
	host := strings.TrimPrefix(strings.ToLower(u.Hostname()), "www.")
	if host != "en.wikipedia.org" || !strings.HasPrefix(u.Path, "/wiki/") {
		return ""
	}
	title, err := url.PathUnescape(strings.TrimPrefix(u.Path, "/wiki/"))
	if err != nil {
		return ""
	}
	title = strings.TrimSpace(strings.ReplaceAll(title, "_", " "))
	if title == "" || strings.Contains(title, ":") {
		return ""
	}
	return title
}

func clipBytes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	s = s[:n]
	for !utf8.ValidString(s) && len(s) > 0 {
		s = s[:len(s)-1]
	}
	return s
}

func (c *Client) getJSON(ctx context.Context, raw string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, raw, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "ModelRun/0.2 (https://ai.encryptid.systems)")
	res, err := c.http().Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	buf, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 || !json.Valid(buf) {
		return nil, errStatus
	}
	return buf, nil
}

var errStatus = errString("lookup")

type errString string

func (e errString) Error() string { return string(e) }

// Lookup runs one named search. Wikipedia and DuckDuckGo use their public
// APIs, then the desk when that read fails.
func (c *Client) Lookup(ctx context.Context, q, source string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(source)) {
	case "wiki", "wikipedia":
		return c.wiki(ctx, q)
	case "duckduckgo", "ddg":
		return c.ddg(ctx, q)
	default:
		return c.Search(ctx, q, source)
	}
}

func (c *Client) wiki(ctx context.Context, q string) (string, error) {
	q = strings.TrimSpace(q)
	if q == "" {
		return "Missing search query.", nil
	}
	if len(q) > 200 {
		q = q[:200]
	}
	api := wikiAPI + "?action=query&list=search&format=json&origin=*&srlimit=5&srnamespace=0&srprop=snippet&srsearch=" + url.QueryEscape(q)
	raw, err := c.getJSON(ctx, api)
	if err != nil {
		return c.Search(ctx, q, "wiki")
	}
	found := related(q, parseWikiSearch(raw))
	if len(found) == 0 {
		out, err := c.Search(ctx, q, "wiki")
		found = related(q, parseHits(out))
		if len(found) == 0 {
			if err != nil {
				return "The lookup failed.", err
			}
			return "Nothing matched.", nil
		}
	}
	return formatHits(found), nil
}

func (c *Client) ddg(ctx context.Context, q string) (string, error) {
	q = strings.TrimSpace(q)
	if q == "" {
		return "Missing search query.", nil
	}
	api := "https://api.duckduckgo.com/?q=" + url.QueryEscape(q) + "&format=json&no_html=1&no_redirect=1&skip_disambig=1"
	var hits []hit
	if raw, err := c.getJSON(ctx, api); err == nil {
		hits = related(q, parseDdgInstant(raw))
	}
	if len(hits) == 0 {
		out, err := c.Search(ctx, q, "duckduckgo")
		hits = related(q, parseHits(out))
		if len(hits) == 0 {
			if err != nil {
				return "The lookup failed.", err
			}
			return "Nothing matched.", nil
		}
	}
	return formatHits(hits), nil
}

// Read loads one page. A Wikipedia article comes from the extract API.
func (c *Client) Read(ctx context.Context, raw string) (string, error) {
	if title := wikiTitle(raw); title != "" {
		api := wikiAPI + "?action=query&prop=extracts&explaintext=1&exsectionformat=plain&redirects=1&format=json&origin=*&titles=" + url.QueryEscape(title)
		if body, err := c.getJSON(ctx, api); err == nil {
			if text := parseWikiExtract(body); text != "" {
				return text, nil
			}
		}
	}
	return c.Gather(ctx, raw)
}
