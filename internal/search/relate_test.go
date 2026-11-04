package search

import "testing"

func TestRelatedDropsStrayPages(t *testing.T) {
	hits := []hit{
		{title: "Capital punishment in France", url: "https://en.wikipedia.org/wiki/Capital_punishment_in_France", snippet: "Capital punishment in France is banned."},
		{title: "List of capitals of France", url: "https://en.wikipedia.org/wiki/List_of_capitals_of_France", snippet: "The capital of France has been Paris since 1944."},
		{title: "jstor.org", url: "https://www.jstor.org/action/doBasicSearch?Query=France", snippet: "List of capitals of France"},
	}
	got := related("the capital of France", hits)
	if len(got) != 1 || got[0].title != "List of capitals of France" {
		t.Fatalf("%#v", got)
	}
}

func TestRelatedDropsSharedNoun(t *testing.T) {
	hits := []hit{
		{title: "Taco Bell menu", url: "https://en.wikipedia.org/wiki/Taco_Bell", snippet: "The menu is large and changes often."},
		{title: "The Cheesecake Factory", url: "https://en.wikipedia.org/wiki/The_Cheesecake_Factory", snippet: "The menu lists hundreds of items."},
	}
	got := related("Why is the cheesecake factory menu so large?", hits)
	if len(got) != 1 || got[0].title != "The Cheesecake Factory" {
		t.Fatalf("%#v", got)
	}
	if CarriesName("Why is the cheesecake factory menu so large?", "The Taco Bell menu is large because it changes often.") {
		t.Fatal("taco bell")
	}
}

func TestRelatedNewsKeepsTheTopic(t *testing.T) {
	hits := []hit{
		{title: "Baking bread at home", url: "https://example.com/bread", snippet: "A recipe for bread."},
		{title: "Ukraine latest", url: "https://www.reuters.com/world/ukraine", snippet: "The latest news on Ukraine."},
		{title: "World news", url: "https://example.com/world", snippet: "Headlines from many countries."},
	}
	got := related("what is the latest news about Ukraine", hits)
	if len(got) != 1 || got[0].title != "Ukraine latest" {
		t.Fatalf("%#v", got)
	}
}
