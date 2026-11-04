package search

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"modelrun/internal/agent"
)

const pageChars = 4500

type item struct {
	kind  string
	tool  string
	label string
	text  string
	url   string
	kept  bool
	page  bool
}

func sourceKept(output string) bool {
	s := strings.Join(strings.Fields(output), " ")
	if len(s) < 12 {
		return false
	}
	low := strings.ToLower(s)
	for _, p := range []string{
		"the lookup failed",
		"missing search query",
		"nothing matched",
		"unknown tool",
		"that address is not available",
	} {
		if strings.HasPrefix(low, p) {
			return false
		}
	}
	if strings.Contains(strings.ToLower(s), "reads only an address the person gave or a search listed") {
		return false
	}
	return true
}

var navLead = regexp.MustCompile(`(?i)^(?:(?:skip to content|where to eat|overview|recipe|comments|home|menu|search|navigation)\s+){2,}`)

func pageProse(text string) string {
	raw := strings.TrimSpace(strings.ReplaceAll(text, "\r", ""))
	flat := strings.Join(strings.Fields(raw), " ")
	if flat == "" {
		return ""
	}
	var kept []string
	dropped := false
	for _, part := range splitSentences(flat) {
		s := strings.TrimSpace(part)
		if s == "" {
			continue
		}
		if chromeRE.MatchString(s) {
			dropped = true
			continue
		}
		kept = append(kept, s)
	}
	body := strings.TrimSpace(strings.Join(kept, " "))
	if dropped {
		body = strings.TrimSpace(navLead.ReplaceAllString(body, ""))
	}
	words := 0
	if body != "" {
		words = len(strings.Fields(body))
	}
	if dropped && words < 12 {
		return ""
	}
	if !dropped {
		if len(flat) < 12 {
			return ""
		}
		return raw
	}
	return body
}

func frameEcho(output, question string) bool {
	q := strings.Join(strings.Fields(question), " ")
	if len(q) < 8 {
		return false
	}
	s := strings.TrimSpace(output)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		s = strings.TrimSpace(s[1 : len(s)-1])
	}
	s = strings.Join(strings.Fields(s), " ")
	if s == "" {
		return false
	}
	a := strings.ToLower(s)
	b := strings.ToLower(q)
	if a == b {
		return true
	}
	if !strings.Contains(a, b) {
		return false
	}
	extra := len(a) - len(b)
	return extra < 160 && extra < len(b)
}

func clipLines(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\r", ""))
	if len(s) <= n {
		return s
	}
	return clipBytes(s, n)
}

func parseHits(text string) []hit {
	src := strings.TrimSpace(strings.ReplaceAll(text, "\r", ""))
	if src == "" {
		return nil
	}
	chunks := regexp.MustCompile(`\n{2,}`).Split(src, -1)
	num := regexp.MustCompile(`^\d{1,2}[.)]\s+`)
	var good []string
	for i, c := range chunks {
		if num.MatchString(strings.TrimSpace(c)) {
			good = append(good, c)
			continue
		}
		if i == len(chunks)-1 {
			break
		}
		return nil
	}
	chunks = good
	if len(chunks) == 0 {
		return nil
	}
	var hits []hit
	for _, chunk := range chunks {
		var lines []string
		for _, line := range strings.Split(chunk, "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				lines = append(lines, line)
			}
		}
		if len(lines) == 0 {
			continue
		}
		title := num.ReplaceAllString(lines[0], "")
		u := ""
		for _, line := range lines {
			if strings.HasPrefix(line, "http://") || strings.HasPrefix(line, "https://") {
				u = line
				break
			}
		}
		var snip []string
		for _, line := range lines {
			if line == lines[0] || line == u {
				continue
			}
			snip = append(snip, line)
		}
		if title == "" {
			continue
		}
		hits = append(hits, hit{title: title, url: u, snippet: strings.Join(snip, " ")})
	}
	return hits
}

func wordsOf(s string) []string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else {
			b.WriteByte(' ')
		}
	}
	return strings.Fields(b.String())
}

var citeStop = map[string]bool{}

func init() {
	for _, w := range strings.Fields("the a an of and to in on for is was are were what who where when why how me about look up lookup tell please with from that this into its or as at by be it") {
		citeStop[w] = true
	}
}

func queryWords(question string) []string {
	var out []string
	for _, w := range wordsOf(question) {
		if len(w) > 2 && !citeStop[w] {
			out = append(out, w)
		}
	}
	return out
}

// OffSubject is a short reply that shares no content word with the question.
// "the suitcase is too small" does not answer a question about a menu.
func OffSubject(question, text string) bool {
	words := queryWords(question)
	if len(words) == 0 {
		return false
	}
	text = strings.TrimSpace(text)
	if text == "" || len(strings.Fields(text)) > 24 {
		return false
	}
	return scoreSentence(text, words) == 0
}

func scoreSentence(sentence string, qwords []string) int {
	if len(qwords) == 0 {
		return 1
	}
	bag := map[string]bool{}
	for _, w := range wordsOf(sentence) {
		bag[w] = true
	}
	n := 0
	for _, w := range qwords {
		if bag[w] {
			n++
		}
	}
	return n
}

var quoteLead = regexp.MustCompile(`(?i)^(?:home|menu|search|cookie|sign in|subscribe|share|advertisement|i will|i have generated|let me know if|please note)\b`)

func quoteSentences(text string, minWords, maxSentences int) []string {
	if minWords <= 0 {
		minWords = 8
	}
	if maxSentences <= 0 {
		maxSentences = 16
	}
	flat := strings.Join(strings.Fields(text), " ")
	var out []string
	for _, part := range splitSentences(flat) {
		s := strings.TrimSpace(part)
		n := len(strings.Fields(s))
		if n < minWords || n > 120 {
			continue
		}
		if !strings.HasSuffix(s, ".") && !strings.HasSuffix(s, "!") && !strings.HasSuffix(s, "?") && !strings.HasSuffix(s, `."`) && !strings.HasSuffix(s, `?"`) && !strings.HasSuffix(s, `!"`) {
			continue
		}
		if quoteLead.MatchString(s) {
			continue
		}
		out = append(out, s)
		if len(out) >= maxSentences {
			break
		}
	}
	return out
}

func leadQuotes(text string, n int) []string {
	quotes := quoteSentences(text, 5, n)
	if len(quotes) > 0 {
		if len(quotes) > n {
			quotes = quotes[:n]
		}
		return quotes
	}
	var out []string
	for _, chunk := range regexp.MustCompile(`\n{2,}`).Split(text, -1) {
		s := strings.Join(strings.Fields(chunk), " ")
		if len(strings.Fields(s)) < 8 {
			continue
		}
		if len(s) > 500 {
			s = clipBytes(s, 500)
			if i := strings.LastIndex(s, " "); i > 0 {
				s = strings.TrimSpace(s[:i])
			}
		}
		out = append(out, s)
		if len(out) >= n {
			break
		}
	}
	return out
}

func pickSentences(text string, qwords []string, n int, lead bool) []string {
	sentences := leadQuotes(text, n)
	if len(sentences) == 0 {
		return nil
	}
	if lead {
		if len(sentences) > n {
			return sentences[:n]
		}
		return sentences
	}
	type row struct {
		s     string
		i     int
		score int
	}
	ranked := make([]row, len(sentences))
	for i, s := range sentences {
		ranked[i] = row{s: s, i: i, score: scoreSentence(s, qwords)}
	}
	for i := 1; i < len(ranked); i++ {
		j := i
		for j > 0 && (ranked[j].score > ranked[j-1].score || (ranked[j].score == ranked[j-1].score && ranked[j].i < ranked[j-1].i)) {
			ranked[j], ranked[j-1] = ranked[j-1], ranked[j]
			j--
		}
	}
	var best []row
	for _, row := range ranked {
		if row.score <= 0 {
			continue
		}
		best = append(best, row)
		if len(best) >= n {
			break
		}
	}
	if len(best) == 0 {
		return nil
	}
	for i := 1; i < len(best); i++ {
		j := i
		for j > 0 && best[j].i < best[j-1].i {
			best[j], best[j-1] = best[j-1], best[j]
			j--
		}
	}
	out := make([]string, len(best))
	for i, row := range best {
		out[i] = row.s
	}
	return out
}

func pageItems(trace []item) []item {
	var out []item
	for _, it := range trace {
		if it.kind == "tool" && it.page && it.kept && it.text != "" {
			out = append(out, it)
		}
	}
	return out
}

func citedSources(trace []item) []hit {
	var out []hit
	seen := map[string]bool{}
	for _, it := range pageItems(trace) {
		if it.url == "" || seen[it.url] {
			continue
		}
		seen[it.url] = true
		title := it.label
		if title == "" {
			title = it.url
		}
		out = append(out, hit{title: title, url: it.url})
		if len(out) >= 5 {
			return out
		}
	}
	if len(out) > 0 {
		return out
	}
	for _, it := range trace {
		if it.kind != "tool" || !it.kept || it.text == "" {
			continue
		}
		for _, h := range parseHits(it.text) {
			if h.url == "" || seen[h.url] {
				continue
			}
			seen[h.url] = true
			out = append(out, h)
			if len(out) >= 5 {
				return out
			}
		}
	}
	return out
}

func groundHits(question string, trace []item) []hit {
	qwords := queryWords(question)
	pages := pageItems(trace)
	sources := citedSources(trace)
	if len(pages) == 0 || len(qwords) == 0 {
		return sources
	}
	byURL := map[string]item{}
	for _, page := range pages {
		byURL[page.url] = page
	}
	best := 0
	scores := make([]int, len(sources))
	for i, src := range sources {
		text := src.snippet
		if page, ok := byURL[src.url]; ok && page.text != "" {
			text = page.text
		}
		scores[i] = scoreSentence(text, qwords)
		if scores[i] > best {
			best = scores[i]
		}
	}
	if best < 3 {
		return sources
	}
	var kept []hit
	for i, src := range sources {
		if scores[i] >= best-1 {
			kept = append(kept, src)
		}
	}
	if len(kept) == 0 {
		return sources
	}
	return kept
}

type pick struct {
	s   string
	n   int
	url string
}

func pageAnswer(question string, trace []item) string {
	qwords := queryWords(question)
	sources := groundHits(question, trace)
	pages := pageItems(trace)
	var picks []pick
	for _, it := range trace {
		if len(picks) >= 6 {
			break
		}
		if it.kind != "tool" || it.tool != "search" || !it.kept || it.text == "" {
			continue
		}
		label := strings.ToLower(it.label)
		if !strings.Contains(label, "duck.ai") && !strings.Contains(label, "gemini") {
			continue
		}
		if len(parseHits(it.text)) > 0 {
			continue
		}
		if scoreSentence(it.text, qwords) < 2 {
			continue
		}
		if shaped := formatRecipe(it.text); shaped != "" {
			picks = append(picks, pick{s: shaped})
			continue
		}
		for _, s := range templateQuotes(question, it.text, 6) {
			if len(picks) >= 6 || containsPick(picks, s) {
				continue
			}
			picks = append(picks, pick{s: s})
		}
	}
	if len(pages) > 0 && len(sources) > 0 {
		byURL := map[string]item{}
		for _, page := range pages {
			byURL[page.url] = page
		}
		ordered := append([]hit(nil), sources...)
		for i := 1; i < len(ordered); i++ {
			j := i
			for j > 0 {
				as := scoreSentence(pageText(byURL, ordered[j]), qwords)
				bs := scoreSentence(pageText(byURL, ordered[j-1]), qwords)
				if as <= bs {
					break
				}
				ordered[j], ordered[j-1] = ordered[j-1], ordered[j]
				j--
			}
		}
		for i, src := range ordered {
			page, ok := byURL[src.url]
			if !ok || len(picks) >= 12 {
				continue
			}
			n := indexHit(sources, src.url) + 1
			if shaped := formatRecipe(page.text); shaped != "" {
				picks = append(picks, pick{s: shaped, n: n, url: src.url})
				continue
			}
			limit := 2
			lead := false
			if i == 0 {
				limit = 8
				lead = true
			}
			for _, s := range pickSentences(page.text, qwords, limit, lead) {
				if len(picks) >= 12 || containsPick(picks, s) {
					continue
				}
				picks = append(picks, pick{s: s, n: n, url: src.url})
			}
		}
	}
	if len(picks) == 0 && len(sources) > 0 {
		for i, src := range sources {
			got := pickSentences(src.snippet, qwords, 1, true)
			if len(got) == 0 || len(picks) >= 4 || containsPick(picks, got[0]) {
				continue
			}
			picks = append(picks, pick{s: got[0], n: i + 1, url: src.url})
		}
	}
	if len(picks) > 0 {
		return applyTemplate(question, picks)
	}
	var quotes []string
	for _, it := range trace {
		if it.kind != "tool" || !it.kept || it.text == "" {
			continue
		}
		for _, s := range leadQuotes(it.text, 2) {
			if !containsString(quotes, s) {
				quotes = append(quotes, s)
			}
		}
		if len(quotes) > 0 {
			break
		}
	}
	out := make([]pick, len(quotes))
	for i, s := range quotes {
		out[i] = pick{s: s}
	}
	return applyTemplate(question, out)
}

func pageText(byURL map[string]item, src hit) string {
	if page, ok := byURL[src.url]; ok {
		return page.text
	}
	return ""
}

func indexHit(sources []hit, rawURL string) int {
	for i, src := range sources {
		if src.url == rawURL {
			return i
		}
	}
	return 0
}

func containsPick(picks []pick, s string) bool {
	for _, p := range picks {
		if p.s == s {
			return true
		}
	}
	return false
}

func containsString(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func citeLine(sentence string, n int, rawURL string) string {
	if rawURL == "" {
		return sentence
	}
	return sentence + " [" + strconv.Itoa(n) + "](" + rawURL + ")"
}

// Brief is the lookup text inside a grounded prompt. A reply that is only
// the word fact is replaced with this.
func Brief(prompt string) string {
	const mark = "Notes:\n"
	i := strings.Index(prompt, mark)
	if i < 0 {
		line, _, _ := strings.Cut(prompt, "\n")
		return strings.TrimSpace(line)
	}
	rest := prompt[i+len(mark):]
	if j := strings.LastIndex(rest, "\nQuestion:"); j >= 0 {
		rest = rest[:j]
	}
	return strings.TrimSpace(rest)
}

// modelBrief is what the model should say the answer from. The notes are a
// few sentences from the pages, not the pages themselves.
func modelBrief(question string, trace []item) (prompt, sources string) {
	notes := briefNotes(question, trace)
	sources = strings.TrimSpace(withSources("", trace, question))
	if len(notes) == 0 {
		return "The lookup returned nothing on this. Say that in one sentence.\nQuestion: " + question, ""
	}
	return "The lookup is already done. Answer the question the person asked, in plain words. " +
		"Use the notes. Give the answer that question asks for. Do not answer with a label, do not use a heading, and do not use JSON.\nNotes:\n" +
		strings.Join(notes, "\n") + "\nQuestion: " + question, sources
}

func briefNotes(question string, trace []item) []string {
	qwords := queryWords(question)
	byURL := map[string]item{}
	for _, page := range pageItems(trace) {
		byURL[page.url] = page
	}
	var notes []string
	seen := map[string]bool{}
	add := func(label, text string, n int, lead bool) {
		if text == "" || len(notes) >= 3 {
			return
		}
		sents := pickSentences(text, qwords, n, lead)
		if len(sents) == 0 {
			return
		}
		blob := strings.Join(sents, " ")
		if len(blob) > 480 {
			blob = clipBytes(blob, 480)
		}
		key := strings.ToLower(blob)
		if seen[key] {
			return
		}
		seen[key] = true
		if label == "" {
			label = "Page"
		}
		notes = append(notes, label+": "+blob)
	}
	first := true
	for _, src := range groundHits(question, trace) {
		page, ok := byURL[src.url]
		if !ok {
			continue
		}
		n := 2
		if first {
			n = 4
		}
		add(page.label, page.text, n, first)
		first = false
		if len(notes) >= 2 {
			break
		}
	}
	if len(notes) > 0 {
		return notes
	}
	for _, it := range trace {
		if it.kind != "tool" || it.tool != "search" || !it.kept {
			continue
		}
		for _, h := range parseHits(it.text) {
			if h.snippet == "" {
				continue
			}
			add(h.title, h.snippet, 1, true)
			if len(notes) >= 3 {
				return notes
			}
		}
	}
	return notes
}

func withSources(answer string, trace []item, question string) string {
	hits := citedSources(trace)
	if question != "" {
		hits = groundHits(question, trace)
	}
	body := strings.TrimSpace(answer)
	if i := regexp.MustCompile(`(?i)\n*Sources:[\s\S]*$`).FindStringIndex(body); i != nil {
		body = strings.TrimSpace(body[:i[0]])
	}
	if len(hits) == 0 {
		return body
	}
	var lines []string
	for i, h := range hits {
		title := strings.NewReplacer("[", "", "]", "").Replace(h.title)
		if title == "" {
			title = h.url
		}
		lines = append(lines, strconv.Itoa(i+1)+". ["+title+"]("+h.url+")")
	}
	return body + "\n\nSources:\n" + strings.Join(lines, "\n")
}

var (
	recipeQ   = regexp.MustCompile(`(?i)\b(?:recipes?)\b|\bhow to (?:make|cook|bake|fry|roast|prepare)\b`)
	compareQ  = regexp.MustCompile(`(?i)\b(?:compare|comparison|difference between|versus)\b|\bvs\.?\b`)
	stepsQ    = regexp.MustCompile(`(?i)\b(?:how to|how do i|how can i|steps|instructions)\b`)
	listQ     = regexp.MustCompile(`(?i)\b(?:list|name the|what are the|which are|examples of)\b`)
	defineQ   = regexp.MustCompile(`(?i)\b(?:who|what|when|where)\b|\b(?:tell me about|describe)\b`)
	causeLead = regexp.MustCompile(`(?i)^(?:causes?|effects?|reasons?)\s+`)
)

func answerTemplate(question string) string {
	switch {
	case recipeQ.MatchString(question):
		return "recipe"
	case compareQ.MatchString(question):
		return "compare"
	case stepsQ.MatchString(question):
		return "steps"
	case listQ.MatchString(question):
		return "list"
	case defineQ.MatchString(question):
		return "define"
	default:
		return "prose"
	}
}

func subjectLine(question string) string {
	s := strings.TrimSpace(causeLead.ReplaceAllString(agent.SearchQuery(question), ""))
	if s == "" {
		return ""
	}
	r, size := utf8.DecodeRuneInString(s)
	if r == utf8.RuneError {
		return s
	}
	return string(unicode.ToUpper(r)) + s[size:]
}

func headed(question, body string) string {
	title := subjectLine(question)
	if title == "" {
		return body
	}
	return "## " + title + "\n\n" + body
}

func templateQuotes(question, text string, n int) []string {
	kind := answerTemplate(question)
	if kind == "steps" || kind == "list" || kind == "compare" {
		var parts []string
		for _, s := range splitSentences(strings.Join(strings.Fields(text), " ")) {
			s = strings.TrimSpace(s)
			if len(wordsOf(s)) < 3 {
				continue
			}
			parts = append(parts, s)
			if len(parts) >= n {
				break
			}
		}
		if len(parts) > 0 {
			return parts
		}
	}
	return leadQuotes(text, n)
}

type block struct {
	s      string
	line   string
	shaped bool
}

func applyTemplate(question string, picks []pick) string {
	if len(picks) == 0 {
		return ""
	}
	blocks := make([]block, len(picks))
	for i, p := range picks {
		shaped := regexp.MustCompile(`(?m)^\s*##\s`).MatchString(p.s) || strings.Contains(p.s, "| Ingredient |")
		blocks[i] = block{s: p.s, line: citeLine(p.s, p.n, p.url), shaped: shaped}
	}
	switch answerTemplate(question) {
	case "recipe":
		var shaped []string
		for _, b := range blocks {
			if b.shaped {
				shaped = append(shaped, b.line)
			}
		}
		if len(shaped) > 0 {
			return strings.Join(shaped, "\n\n")
		}
		return headed(question, numbered(blocks))
	case "compare":
		if table := compareTable(question, blocks); table != "" {
			return table
		}
	case "steps":
		return headed(question, numbered(blocks))
	case "list":
		return headed(question, bullets(blocks))
	case "define":
		lines := make([]string, len(blocks))
		for i, b := range blocks {
			lines[i] = b.line
		}
		return headed(question, strings.Join(lines, "\n\n"))
	}
	lines := make([]string, len(blocks))
	for i, b := range blocks {
		lines[i] = b.line
	}
	return strings.Join(lines, "\n\n")
}

func numbered(blocks []block) string {
	var b strings.Builder
	i := 0
	for n, blk := range blocks {
		if n > 0 {
			b.WriteByte('\n')
		}
		if blk.shaped {
			b.WriteString(blk.line)
			continue
		}
		i++
		b.WriteString(strconv.Itoa(i))
		b.WriteString(". ")
		b.WriteString(blk.line)
	}
	return b.String()
}

func bullets(blocks []block) string {
	var b strings.Builder
	for i, blk := range blocks {
		if i > 0 {
			b.WriteByte('\n')
		}
		if blk.shaped {
			b.WriteString(blk.line)
			continue
		}
		b.WriteString("- ")
		b.WriteString(blk.line)
	}
	return b.String()
}

func sideWords(side string) []string {
	var out []string
	for _, w := range wordsOf(side) {
		if len(w) > 2 && !citeStop[w] {
			out = append(out, w)
		}
	}
	return out
}

func mentionsSide(sentence, side string) bool {
	words := sideWords(side)
	if len(words) == 0 {
		return false
	}
	bag := map[string]bool{}
	for _, w := range wordsOf(sentence) {
		bag[w] = true
	}
	n := 0
	for _, w := range words {
		if bag[w] {
			n++
		}
	}
	need := 2
	if len(words) < need {
		need = len(words)
	}
	return n >= need
}

func compareSides(question string) (string, string, bool) {
	t := strings.TrimRight(strings.Join(strings.Fields(question), " "), "?.! ")
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)\bdifference between\s+(.+?)\s+and\s+(.+)$`),
		regexp.MustCompile(`(?i)\bcompare\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+)$`),
		regexp.MustCompile(`(?i)\b([A-Za-z][^?]{0,48}?)\s+(?:versus|vs\.?)\s+([^?]+)$`),
	}
	for _, re := range patterns {
		m := re.FindStringSubmatch(t)
		if len(m) != 3 {
			continue
		}
		a := strings.TrimSpace(regexp.MustCompile(`(?i)^(?:the|a|an)\s+`).ReplaceAllString(m[1], ""))
		b := strings.TrimSpace(regexp.MustCompile(`(?i)^(?:the|a|an)\s+`).ReplaceAllString(m[2], ""))
		if a != "" && b != "" && len(a) < 80 && len(b) < 80 {
			return a, b, true
		}
	}
	return "", "", false
}

func compareTable(question string, blocks []block) string {
	leftSide, rightSide, ok := compareSides(question)
	if !ok {
		return ""
	}
	var left, right, rest []string
	cell := func(s string) string {
		return strings.Join(strings.Fields(strings.ReplaceAll(s, "|", "/")), " ")
	}
	for _, b := range blocks {
		if b.shaped {
			rest = append(rest, b.line)
			continue
		}
		onLeft := mentionsSide(b.s, leftSide)
		onRight := mentionsSide(b.s, rightSide)
		switch {
		case onLeft && !onRight:
			left = append(left, b.line)
		case onRight && !onLeft:
			right = append(right, b.line)
		default:
			rest = append(rest, b.line)
		}
	}
	if len(left) == 0 || len(right) == 0 {
		return ""
	}
	n := len(left)
	if len(right) > n {
		n = len(right)
	}
	var rows []string
	for i := 0; i < n; i++ {
		l, r := "", ""
		if i < len(left) {
			l = left[i]
		}
		if i < len(right) {
			r = right[i]
		}
		rows = append(rows, "| "+cell(l)+" | "+cell(r)+" |")
	}
	table := "| " + cell(leftSide) + " | " + cell(rightSide) + " |\n| --- | --- |\n" + strings.Join(rows, "\n")
	body := table
	if len(rest) > 0 {
		body += "\n\n" + strings.Join(rest, "\n\n")
	}
	return headed(question, body)
}

func splitSentences(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r != '.' && r != '!' && r != '?' {
			i += size
			continue
		}
		j := i + size
		k := j
		for k < len(s) {
			sp, n := utf8.DecodeRuneInString(s[k:])
			if !unicode.IsSpace(sp) {
				break
			}
			k += n
		}
		if k == j {
			i += size
			continue
		}
		out = append(out, s[start:j])
		start = k
		i = k
	}
	if start < len(s) && strings.TrimSpace(s[start:]) != "" {
		out = append(out, s[start:])
	}
	return out
}
