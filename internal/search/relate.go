package search

import "strings"

// genericWord is a word that does not name the subject. "latest news about
// Ukraine" is about Ukraine, not about the word news.
var genericWord = map[string]bool{
	"news": true, "latest": true, "today": true, "tonight": true,
	"headline": true, "headlines": true, "breaking": true, "current": true,
	"update": true, "updates": true, "weather": true, "forecast": true,
	"right": true, "now": true, "price": true, "prices": true, "cost": true,
	"score": true, "scores": true, "standing": true, "standings": true,
}

func subjectWords(q string) []string {
	var out []string
	for _, w := range queryWords(q) {
		if genericWord[w] {
			continue
		}
		out = append(out, w)
	}
	return out
}

// nameWords is the one or two longest subject words. "menu" and "large" are
// shorter than "cheesecake" and "factory", so a Taco Bell menu page is not
// that subject.
func nameWords(words []string) []string {
	if len(words) <= 2 {
		return words
	}
	first, second := 0, 1
	if len(words[1]) > len(words[0]) {
		first, second = 1, 0
	}
	for i := 2; i < len(words); i++ {
		if len(words[i]) > len(words[first]) {
			second = first
			first = i
		} else if len(words[i]) > len(words[second]) {
			second = i
		}
	}
	return []string{words[first], words[second]}
}

// CarriesName reports that text contains the name in the question.
func CarriesName(question, text string) bool {
	names := nameWords(subjectWords(question))
	if len(names) == 0 {
		return true
	}
	return covers(text, names)
}

func covers(text string, words []string) bool {
	toks := wordsOf(text)
	for _, w := range words {
		if !tokenHas(toks, w) {
			return false
		}
	}
	return true
}

func stemMatch(tok, word string) bool {
	if tok == word {
		return true
	}
	if len(word) >= 4 && strings.HasPrefix(tok, word) && len(tok) <= len(word)+2 {
		return true
	}
	if len(tok) >= 4 && strings.HasPrefix(word, tok) && len(word) <= len(tok)+2 {
		return true
	}
	return false
}

// hasPhrase reports whether words occur in order with at most one token between each pair.
func hasPhrase(text string, words []string) bool {
	if len(words) == 0 {
		return false
	}
	toks := wordsOf(text)
	for i := 0; i < len(toks); i++ {
		if !stemMatch(toks[i], words[0]) {
			continue
		}
		at := i
		ok := true
		for _, w := range words[1:] {
			found := -1
			limit := at + 2
			if limit >= len(toks) {
				limit = len(toks) - 1
			}
			for j := at + 1; j <= limit; j++ {
				if stemMatch(toks[j], w) {
					found = j
					break
				}
			}
			if found < 0 {
				ok = false
				break
			}
			at = found
		}
		if ok {
			return true
		}
	}
	return false
}

// related keeps at most two hits that are about the query. A closer phrase
// match drops a page that only shares the words in a different order.
func related(q string, hits []hit) []hit {
	var clean []hit
	for _, h := range hits {
		if h.title == "" || h.url == "" || skipPage(h.url) {
			continue
		}
		clean = append(clean, h)
	}
	words := subjectWords(q)
	if len(words) == 0 {
		if len(clean) > 2 {
			clean = clean[:2]
		}
		return clean
	}
	phrase := false
	score := make([]int, len(clean))
	best := 0
	for i, h := range clean {
		blob := h.title + " " + h.snippet
		s := 0
		if hasPhrase(blob, words) {
			s += 5
			phrase = true
		}
		title := wordsOf(h.title)
		snip := wordsOf(h.snippet)
		for _, w := range words {
			if tokenHas(title, w) {
				s += 2
			} else if tokenHas(snip, w) {
				s += 1
			}
		}
		score[i] = s
		if s > best {
			best = s
		}
	}
	names := nameWords(words)
	var out []hit
	for i, h := range clean {
		if !covers(h.title+" "+h.snippet, names) {
			continue
		}
		if phrase && score[i] < 5 {
			continue
		}
		if !phrase && (score[i] < 2 || score[i] < best-1) {
			continue
		}
		out = append(out, h)
		if len(out) == 2 {
			break
		}
	}
	return out
}

func tokenHas(toks []string, word string) bool {
	for _, tok := range toks {
		if stemMatch(tok, word) {
			return true
		}
	}
	return false
}
