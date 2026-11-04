package agent

import "strings"

// Route is what a question is, from a think pass. It is not read off the
// person's wording.
type Route string

const (
	// Fact is a checkable fact a reference page can settle.
	Fact Route = "fact"
	// Recipe is a dish. The model writes the ingredients and the method.
	Recipe Route = "recipe"
	// Command is a command or a file listing in the guest.
	Command Route = "command"
	// Talk is everything the model should answer itself.
	Talk Route = "talk"
)

// ThinkSystem is the think pass. The model reflects on the question and
// replies with one route word. The person's phrasing is not a pattern list.
const ThinkSystem = "Think about what this question is asking for. Do not answer it. " +
	"Reply with one word and nothing else.\n" +
	"fact — a checkable fact a reference page can settle, such as who holds an office, a capital, a date, a price, the news, or the weather.\n" +
	"recipe — they want to cook a dish. You know the ingredients and the method. A page about the dish is the author's story, not the recipe.\n" +
	"command — they want a command run, or a file listed, in the guest.\n" +
	"talk — you answer it yourself: a meaning, an opinion, a story, code, a greeting, math, or anything that is not a checkable fact."

// RouteOnly is a reply that is nothing but a route word, such as "Recipe."
// or "Fact". That word names the question. It is not the answer.
func RouteOnly(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" || strings.Contains(s, "\n") {
		return false
	}
	_, ok := labelLine(s)
	return ok
}

// RecipeSystem is the answer pass for a dish. It asks for the method.
const RecipeSystem = "Write the ingredients and the method. Use plain sentences. Start with the ingredients. A label is not the answer."

// RecipeRetry is the second pass when the first reply was only the word recipe.
const RecipeRetry = "List the ingredients, then the steps. Start with the first ingredient."

// FactSystem is the answer pass after a lookup. The device system says to
// keep the answer short and to stop in one sentence when unsure. That is
// how the reply never answers the question that was asked.
const FactSystem = "Answer the question the person asked. Use the notes. A label is not the answer."

// Named reports that the think pass put a route word on its own line.
func Named(thought string) bool {
	for _, line := range strings.Split(thought, "\n") {
		if _, ok := labelLine(line); ok {
			return true
		}
	}
	return false
}

// FactWhenUnnamed is a checkable question whose think pass never named a
// route. A line that names talk, recipe, command, or fact stands. A recipe
// stays with the model: a page about the dish is not the method. A model
// that answers the question instead of naming a route still gets the lookup.
func FactWhenUnnamed(question, thought string) bool {
	if Named(thought) {
		return false
	}
	q := strings.TrimSpace(question)
	if recipe.MatchString(q) {
		return false
	}
	return NeedsFacts(q)
}

// RouteOf reads the route word from a think pass. A reflection that never
// names one of the four words is talk.
func RouteOf(thought string) Route {
	for _, line := range strings.Split(thought, "\n") {
		if route, ok := labelLine(line); ok {
			return route
		}
	}
	return Talk
}

func labelLine(s string) (Route, bool) {
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) == 2 && strings.EqualFold(strings.Trim(fields[0], ":"), "route") {
		return known(fields[1])
	}
	if len(fields) != 1 {
		return "", false
	}
	return known(fields[0])
}

func known(s string) (Route, bool) {
	switch strings.ToLower(strings.Trim(s, "`*_\"'.:!")) {
	case "fact":
		return Fact, true
	case "recipe":
		return Recipe, true
	case "command":
		return Command, true
	case "talk":
		return Talk, true
	default:
		return "", false
	}
}
