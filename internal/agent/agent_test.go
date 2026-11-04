package agent

import (
	"strings"
	"testing"
)

func TestParseExec(t *testing.T) {
	c := Parse("{\"type\":\"tool_call\",\"tool\":\"exec\",\"args\":{\"cmd\":\"ls -la\"}}")
	if c == nil || c.Tool != "exec" || c.Cmd() != "ls -la" {
		t.Fatalf("%#v", c)
	}
	if Slash("/run ls -la").Cmd() != "ls -la" {
		t.Fatal("slash")
	}
	if name, _, ok := Verb("/tools"); !ok || name != "tools" {
		t.Fatal("tools")
	}
	if name, arg, ok := Verb("/skill recipe"); !ok || name != "skill" || arg != "recipe" {
		t.Fatalf("skill %s %s", name, arg)
	}
	if _, _, ok := Verb("/skills"); !ok {
		t.Fatal("skills")
	}
	if Parse("Sure, here is a recipe.") != nil {
		t.Fatal("prose")
	}
	s := Parse("{\"type\":\"tool_call\",\"tool\":\"search\",\"args\":{\"q\":\"lake champlain\"}}")
	if s == nil || s.Tool != "search" || s.Query() != "lake champlain" {
		t.Fatalf("search %#v", s)
	}
	if !strings.Contains(LookupWire("1. hit"), "Do not use JSON.") {
		t.Fatal("lookup wire")
	}
}
