package search

import (
	"strings"
	"testing"
)

func TestParseWikiSearch(t *testing.T) {
	raw := []byte(`{"query":{"search":[{"title":"Lake Champlain","snippet":"<span class=\"searchmatch\">Lake</span> Champlain is a lake."}]}}`)
	hits := parseWikiSearch(raw)
	if len(hits) != 1 || hits[0].title != "Lake Champlain" {
		t.Fatalf("%#v", hits)
	}
	if hits[0].url != "https://en.wikipedia.org/wiki/Lake_Champlain" {
		t.Fatalf("url %q", hits[0].url)
	}
	if strings.Contains(hits[0].snippet, "<") || !strings.Contains(hits[0].snippet, "Lake Champlain") {
		t.Fatalf("snippet %q", hits[0].snippet)
	}
	text := formatHits(hits)
	if !strings.Contains(text, "1. Lake Champlain") {
		t.Fatalf("hits %q", text)
	}
}

func TestParseWikiExtract(t *testing.T) {
	raw := []byte(`{"query":{"pages":{"1":{"extract":"Paris is the capital and largest city of France.\n\nShort.\n\nThe city stands on the Seine and has been a center of politics for centuries."}}}}`)
	got := parseWikiExtract(raw)
	if !strings.Contains(got, "Paris is the capital") || strings.Contains(got, "Short.") {
		t.Fatalf("%q", got)
	}
}

func TestPageAnswerQuotesThePage(t *testing.T) {
	trace := []item{
		{
			kind: "tool", tool: "search", label: "Wikipedia", kept: true,
			text: "1. France\nhttps://en.wikipedia.org/wiki/France\nFrance is a country in Europe.",
		},
		{
			kind: "tool", tool: "gather", label: "France", page: true, kept: true,
			url:  "https://en.wikipedia.org/wiki/France",
			text: "Paris is the capital and largest city of France. The country sits in Western Europe and has a long political history.",
		},
	}
	got := withSources(pageAnswer("what is the capital of France", trace), trace, "what is the capital of France")
	if !strings.Contains(got, "Paris is the capital") || !strings.Contains(got, "https://en.wikipedia.org/wiki/France") {
		t.Fatalf("%q", got)
	}
}

func TestModelBriefIsSpoken(t *testing.T) {
	trace := []item{
		{
			kind: "tool", tool: "search", label: "Wikipedia", kept: true,
			text: "1. France\nhttps://en.wikipedia.org/wiki/France\nFrance is a country in Europe.",
		},
		{
			kind: "tool", tool: "gather", label: "France", page: true, kept: true,
			url:  "https://en.wikipedia.org/wiki/France",
			text: "Paris is the capital and largest city of France. The country sits in Western Europe and has a long political history.",
		},
	}
	prompt, sources := modelBrief("what is the capital of France", trace)
	if strings.HasPrefix(strings.TrimSpace(prompt), "##") || strings.Contains(prompt, "Sources:") {
		t.Fatalf("prompt is a dump:\n%s", prompt)
	}
	if !strings.Contains(prompt, "plain words") || !strings.Contains(prompt, "Paris is the capital") {
		t.Fatalf("prompt %q", prompt)
	}
	if !strings.Contains(Brief(prompt), "Paris is the capital") || strings.Contains(Brief(prompt), "Question:") {
		t.Fatalf("brief %q", Brief(prompt))
	}
	if !strings.Contains(sources, "https://en.wikipedia.org/wiki/France") {
		t.Fatalf("sources %q", sources)
	}
}

func TestOffSubject(t *testing.T) {
	q := "Why is the cheesecake factory menu so large?"
	if !OffSubject(q, "the suitcase is too small") {
		t.Fatal("suitcase")
	}
	if OffSubject(q, "The Cheesecake Factory menu is long because it lists hundreds of dishes.") {
		t.Fatal("menu")
	}
	if OffSubject("what is 2 + 2", "4") {
		t.Fatal("arithmetic")
	}
}

func TestFormatRecipe(t *testing.T) {
	got := formatRecipe("Fried chicken. Ingredients 2 cups flour 1 tsp salt Method Mix the flour with the salt. Fry the chicken until it is brown.")
	if !strings.Contains(got, "flour") || !strings.Contains(got, "2 cups") || !strings.Contains(got, "salt") {
		t.Fatalf("%q", got)
	}
}
