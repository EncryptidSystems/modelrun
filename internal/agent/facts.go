package agent

import (
	"regexp"
	"strings"
)

// Source is one lookup the fact plan runs. Wiki and DuckDuckGo take the
// subject. duck.ai and Gemini take the question as asked.
type Source struct {
	ID    string
	Label string
}

// GapSources is the fact plan's lookup order.
func GapSources() []Source {
	return []Source{
		{ID: "wiki", Label: "Wikipedia"},
		{ID: "duckduckgo", Label: "DuckDuckGo"},
		{ID: "duck", Label: "duck.ai"},
		{ID: "gemini", Label: "Gemini"},
	}
}

// SubjectSource reports whether id searches the subject rather than the raw question.
func SubjectSource(id string) bool {
	switch strings.ToLower(strings.TrimSpace(id)) {
	case "wiki", "wikipedia", "duckduckgo", "ddg":
		return true
	default:
		return false
	}
}

var (
	plain  = regexp.MustCompile(`(?i)^(hi|hey|hello|thanks|thank you|ok|okay|yes|no|sure|cool|nice|good (?:morning|night|evening))\b[\s!.,?]*$`)
	info   = regexp.MustCompile(`(?i)\b(latest|current|right now|as of|news|headline|weather|forecast|price|cost|score|standing|election|governor|president|population|capital|who is|who was|who are|what is|what was|what are|when did|when was|when is|where is|where was|where are|which|how (?:does|do|did)|why (?:did|was|is|are))\b`)
	recipe = regexp.MustCompile(`(?i)\b(?:recipes?)\b|\bhow (?:do i |to )?(?:make|cook|bake|fry|roast|prepare)\b`)
	// describe is separate: the site's cue excludes "describe how", and Go's
	// regexp engine has no lookahead.
	factCue     = regexp.MustCompile(`(?i)\b(?:search|research|look\s*up|lookup|facts?|causes?|effects?|history|impact)\b|\b(?:provide|cite|with)\s+sources?\b|\b(?:tell me about|write about|essay about)\b`)
	describe    = regexp.MustCompile(`(?i)\bdescribe\b`)
	describeHow = regexp.MustCompile(`(?i)\bdescribe\s+how\b`)
	fiction     = regexp.MustCompile(`(?i)\b(?:story|poem|poems|joke|jokes|haiku|limerick|fiction)\b`)
	codeMath    = regexp.MustCompile(`(?i)\b(?:prove|proof|derive|function|stack trace|exception|compile|regex|sql|segfault|integral|derivative|equation|theorem|probability|algorithm)\b|\d\s*[-+*/^×÷=]\s*\d`)
	arithmetic  = regexp.MustCompile(`(?i)^\s*what(?:'s| is)?\s+\d[\d\s+\-*/^=().]*\??\s*$`)
	aboutSelf   = regexp.MustCompile(`(?i)^\s*(who are you|what(?:'s| is) your name)\b`)
	creative    = regexp.MustCompile(`(?i)^\s*(?:please\s+)?(?:write|tell|compose|make up|create|draft|give me)\b.{0,40}\b(?:story|stories|poem|poems|joke|jokes|haiku|limerick|fiction|song|fable|tale)\b`)
	leadIn      = regexp.MustCompile(`(?i)^(?:(?:please|hey|ok(?:ay)?)[,\s]+)?(?:(?:can|could|would) you\s+)?(?:research|search(?:\s+for)?|look\s*up|lookup|tell me(?:\s+about)?|explain(?:\s+to me)?|describe|give me(?:\s+an?\s+(?:overview|summary)\s+of)?|i (?:want|would like) to know(?:\s+about)?|what(?:'s| is| are| was| were)|who(?:'s| is| are| was| were)|when (?:is|was|did|does)|where (?:is|are|was|were)|why (?:is|are|was|were|do|does|did)|how (?:do|does|did|can|to)(?:\s+(?:i|you|we|they|one))?)\s+`)
	howWhatWhy  = regexp.MustCompile(`(?i)^(?:how|what|why)\s+`)
	recipeLead  = regexp.MustCompile(`(?i)^(?:(?:a|an)\s+)?(?:(?:formatted|simple|easy|basic|quick)\s+)?(?:recipes?|instructions?)(?:\s+(?:for|of|to(?:\s+make)?))?\s+`)
	liveCue     = regexp.MustCompile(`(?i)\b(?:news|headlines?|breaking|latest|today|tonight|this morning|right now|weather|forecast|scores?|standings?|prices?|cost)\b`)
)

// Live is a question a encyclopedia page will not answer: news, weather, a price, a score.
func Live(text string) bool {
	return liveCue.MatchString(strings.TrimSpace(text))
}

func factCueMatch(t string) bool {
	if factCue.MatchString(t) {
		return true
	}
	return describe.MatchString(t) && !describeHow.MatchString(t)
}

// InfoGap is a question the checkpoint does not hold: news, a fact, or a recipe.
func InfoGap(text string) bool {
	t := strings.TrimSpace(text)
	if t == "" || plain.MatchString(t) || aboutSelf.MatchString(t) || arithmetic.MatchString(t) || creative.MatchString(t) {
		return false
	}
	return info.MatchString(t) || recipe.MatchString(t)
}

// NeedsFacts is the old phrase list. The think pass decides whether a
// question is a fact. A new wording is not a new pattern.
func NeedsFacts(text string) bool {
	t := strings.TrimSpace(text)
	if t == "" || plain.MatchString(t) || aboutSelf.MatchString(t) || arithmetic.MatchString(t) {
		return false
	}
	if InfoGap(t) {
		return true
	}
	if fiction.MatchString(t) || codeMath.MatchString(t) {
		return false
	}
	return factCueMatch(t)
}

// PlanQuery is the question, collapsed, capped at 180 characters.
func PlanQuery(text string) string {
	return clip(strings.Join(strings.Fields(text), " "), 180)
}

// SearchQuery is the subject a search engine should see.
// "Tell me about the history of the printing press" becomes
// "the history of the printing press".
func SearchQuery(text string) string {
	raw := PlanQuery(text)
	q := raw
	for i := 0; i < 3; i++ {
		next := strings.TrimSpace(howWhatWhy.ReplaceAllString(leadIn.ReplaceAllString(q, ""), ""))
		if next == q {
			break
		}
		q = next
	}
	q = strings.TrimRight(q, " ?.!")
	q = strings.TrimSpace(q)
	q = strings.TrimSpace(recipeLead.ReplaceAllString(q, ""))
	if q == "" {
		q = raw
	}
	return clip(q, 120)
}

func clip(s string, n int) string {
	if n <= 0 {
		return ""
	}
	i := 0
	for j := range s {
		if i == n {
			return s[:j]
		}
		i++
	}
	return s
}
