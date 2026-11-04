package agent

import "testing"

func TestRouteOf(t *testing.T) {
	if got := RouteOf("fact"); got != Fact {
		t.Fatalf("fact %q", got)
	}
	if got := RouteOf("The question asks who holds an office.\nfact"); got != Fact {
		t.Fatalf("last %q", got)
	}
	if got := RouteOf("recipe."); got != Recipe {
		t.Fatalf("recipe %q", got)
	}
	if got := RouteOf("route: command"); got != Command {
		t.Fatalf("command %q", got)
	}
	if got := RouteOf("They want the meaning, not a page."); got != Talk {
		t.Fatalf("prose %q", got)
	}
	if got := RouteOf("I will look this up and search for it"); got != Talk {
		t.Fatalf("phrasing %q", got)
	}
	if got := RouteOf(""); got != Talk {
		t.Fatalf("empty %q", got)
	}
}

func TestFactWhenUnnamed(t *testing.T) {
	if !FactWhenUnnamed("what is the capital of France", "Paris is the capital of France.") {
		t.Fatal("answered fact")
	}
	if FactWhenUnnamed("what is the capital of France", "fact") {
		t.Fatal("named fact")
	}
	if FactWhenUnnamed("what is the capital of France", "talk") {
		t.Fatal("named talk")
	}
	if FactWhenUnnamed("how to make fried chicken", "Preheat the oven.") {
		t.Fatal("recipe")
	}
	if FactWhenUnnamed("write a poem about a dragon", "Once there was a dragon.") {
		t.Fatal("poem")
	}
	if !Named("The question asks who holds an office.\nfact") {
		t.Fatal("named")
	}
	if Named("Paris is the capital.") {
		t.Fatal("prose")
	}
	if !RouteOnly("Recipe.") || !RouteOnly("Fact") {
		t.Fatal("label")
	}
	if RouteOnly("Preheat the oven to 350.") || FactWhenUnnamed("How do I make a banana cream pie", "Preheat the oven.") {
		t.Fatal("method")
	}
}

func TestAlone(t *testing.T) {
	call := Alone("{\"type\":\"tool_call\",\"tool\":\"exec\",\"args\":{\"cmd\":\"ls\"}}")
	if call == nil || call.Tool != "exec" || call.Cmd() != "ls" {
		t.Fatalf("%#v", call)
	}
	if Alone("Paris is the capital.\n{\"type\":\"tool_call\",\"tool\":\"search\",\"args\":{\"q\":\"france\"}}") != nil {
		t.Fatal("prose")
	}
}
