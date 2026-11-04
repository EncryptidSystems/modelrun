package search

import (
	"regexp"
	"strconv"
	"strings"
)

// formatRecipe turns a flattened recipe page into the table the site quotes.
// A page that is not a recipe returns "".
func formatRecipe(text string) string {
	flat := strings.Join(strings.Fields(text), " ")
	ingAt := regexp.MustCompile(`(?i)\bIngredients?\b(?:\s+Amount)?`).FindStringIndex(flat)
	if ingAt == nil {
		return ""
	}
	rest := flat[ingAt[0]:]
	rel := regexp.MustCompile(`(?i)\b(?:Method|Directions|Instructions|Steps|Preparation|Marinate)\b`).FindStringIndex(rest)
	if rel == nil {
		return ""
	}
	methodAt := ingAt[0] + rel[0]
	head := strings.TrimSpace(flat[:ingAt[0]])
	ing := strings.TrimSpace(regexp.MustCompile(`(?i)^\s*Ingredients?\s*(?:Amount\s*)?`).ReplaceAllString(flat[ingAt[0]:methodAt], ""))
	method := strings.TrimSpace(regexp.MustCompile(`(?i)^\s*(?:Method|Directions|Instructions|Steps|Preparation|Marinate)\s*:?\s*`).ReplaceAllString(flat[methodAt:], ""))
	rows := recipeRows(ing)
	if len(rows) < 2 || len(head) > 140 {
		return ""
	}
	for _, row := range rows {
		if len(row[0]) > 80 || len(strings.Fields(row[0])) > 14 {
			return ""
		}
	}
	meta := recipeMeta(head)
	var steps []string
	for _, s := range splitSentences(method) {
		s = strings.TrimSpace(s)
		if s == "" || regexp.MustCompile(`(?i)^[a-z0-9.-]+\.[a-z]{2,}$`).MatchString(s) {
			continue
		}
		steps = append(steps, s)
	}
	cell := func(s string) string { return strings.ReplaceAll(s, "|", "/") }
	var lines []string
	if meta.title != "" {
		lines = append(lines, "## "+meta.title)
	}
	if len(meta.lines) > 0 {
		var bits []string
		for _, l := range meta.lines {
			bits = append(bits, "- "+l)
		}
		lines = append(lines, strings.Join(bits, "\n"))
	}
	var table []string
	table = append(table, "| Ingredient | Amount |", "| --- | --- |")
	for _, row := range rows {
		amt := row[1]
		if amt == "" {
			amt = "—"
		}
		table = append(table, "| "+cell(row[0])+" | "+cell(amt)+" |")
	}
	lines = append(lines, strings.Join(table, "\n"))
	if len(steps) > 0 {
		var methodLines []string
		for i, s := range steps {
			methodLines = append(methodLines, strconv.Itoa(i+1)+". "+s)
		}
		lines = append(lines, "### Method\n"+strings.Join(methodLines, "\n"))
	}
	return strings.Join(lines, "\n\n")
}

type recipeMetaResult struct {
	title string
	lines []string
}

func recipeMeta(head string) recipeMetaResult {
	re := regexp.MustCompile(`(?i)\b(Makes|Serves|Servings|Prep|Cook(?:\s+time)?|Total|Yield)\s*:\s*`)
	marks := re.FindAllStringSubmatchIndex(head, -1)
	if len(marks) == 0 {
		return recipeMetaResult{title: strings.TrimSpace(head)}
	}
	title := strings.TrimRight(strings.TrimSpace(head[:marks[0][0]]), ".?!")
	var lines []string
	for i, mark := range marks {
		end := len(head)
		if i+1 < len(marks) {
			end = marks[i+1][0]
		}
		value := strings.TrimSpace(head[mark[1]:end])
		value = strings.TrimRight(strings.Join(strings.Fields(value), " "), " ·•|")
		label := head[mark[2]:mark[3]]
		label = regexp.MustCompile(`(?i)\s+time$`).ReplaceAllString(label, "")
		if value != "" {
			lines = append(lines, label+": "+value)
		}
	}
	return recipeMetaResult{title: title, lines: lines}
}

var recipeAmount = regexp.MustCompile(`(?i)(?:about\s+)?(?:as needed|\d+[¼½¾⅓⅔⅛⅜⅝⅞]|\d+\s+\d+/\d+|\d+/\d+|[¼½¾⅓⅔⅛⅜⅝⅞]|\d+(?:\.\d+)?)(?:\s*(?:lb|lbs|pounds?|g|kg|oz|ml|cups?|tbsp|tsp|tablespoons?|teaspoons?|inches?|inch))?(?:\s*\([^)]*\))?(?:,\s*divided)?(?:\s+in the pan)?`)

func recipeRows(section string) [][2]string {
	masked := maskNameParens(section)
	matches := recipeAmount.FindAllStringIndex(masked, -1)
	if len(matches) == 0 {
		return nil
	}
	gaps := make([]string, 0, len(matches)+1)
	start := 0
	for _, m := range matches {
		gaps = append(gaps, cleanName(section[start:m[0]]))
		start = m[1]
	}
	gaps = append(gaps, cleanName(section[start:]))
	amountOf := func(m []int) string {
		return strings.Join(strings.Fields(section[m[0]:m[1]]), " ")
	}
	var before [][2]string
	for i, m := range matches {
		if gaps[i] != "" {
			before = append(before, [2]string{gaps[i], amountOf(m)})
		}
	}
	var after [][2]string
	if gaps[0] != "" && len(strings.Fields(gaps[0])) > 5 {
		after = append(after, [2]string{gaps[0], ""})
	}
	for i, m := range matches {
		if gaps[i+1] != "" {
			after = append(after, [2]string{gaps[i+1], amountOf(m)})
		}
	}
	picked := before
	if scoreRows(after) > scoreRows(before) {
		picked = after
	}
	out := make([][2]string, len(picked))
	for i, row := range picked {
		name, amount := liftParenAmount(row[0], row[1])
		out[i] = [2]string{name, amount}
	}
	return out
}

func scoreRows(rows [][2]string) int {
	n := 0
	for _, row := range rows {
		n += rowScore(row[0], row[1])
	}
	return n
}

func rowScore(name, amount string) int {
	words := len(strings.Fields(name))
	unit := regexp.MustCompile(`(?i)\b(?:cups?|tbsp|tsp|teaspoons?|tablespoons?|lb|lbs|oz|ml|g|kg)\b`).MatchString(amount)
	bare := regexp.MustCompile(`^\d+$`).MatchString(amount)
	score := 0
	if unit && words <= 5 {
		score += 2
	}
	if unit && words > 6 {
		score -= 3
	}
	if bare && regexp.MustCompile(`(?i)egg`).MatchString(name) {
		score += 2
	}
	if bare && !regexp.MustCompile(`(?i)egg`).MatchString(name) {
		score -= 2
	}
	if amount == "" && words > 5 {
		score++
	}
	return score
}

func maskNameParens(section string) string {
	re := regexp.MustCompile(`\(([^)]*)\)`)
	amount := regexp.MustCompile(`(?i)^\s*(?:about\s+)?[\d¼½¾⅓⅔⅛./\s-]+(?:lb|lbs|g|kg|oz|ml|cups?)?\s*$`)
	return re.ReplaceAllStringFunc(section, func(full string) string {
		inner := full[1 : len(full)-1]
		if amount.MatchString(inner) {
			return full
		}
		return strings.Repeat(" ", len(full))
	})
}

func liftParenAmount(name, amount string) (string, string) {
	if amount != "" {
		return name, amount
	}
	m := regexp.MustCompile(`\(([^)]*\d[^)]*)\)`).FindStringSubmatchIndex(name)
	if m == nil {
		return name, amount
	}
	inner := name[m[2]:m[3]]
	if len(inner) > 40 {
		return name, amount
	}
	return cleanName(name[:m[0]] + " " + name[m[1]:]), strings.Join(strings.Fields(inner), " ")
}

func cleanName(s string) string {
	s = strings.Trim(s, " \t\r\n,;:.-–—")
	s = strings.ReplaceAll(s, " ,", ",")
	return strings.Join(strings.Fields(s), " ")
}
