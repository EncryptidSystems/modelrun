package agent

import "strings"

// Verb is /tools, /skills, or /skill. /run and /exec stay on Slash.
func Verb(text string) (name, arg string, ok bool) {
	for _, v := range []string{"/tools", "/skills", "/skill"} {
		rest, hit := slashRest(strings.TrimSpace(text), v)
		if !hit {
			continue
		}
		return strings.TrimPrefix(v, "/"), rest, true
	}
	return "", "", false
}

// ToolsText is the tool list. Each one runs.
func ToolsText() string {
	return strings.Join([]string{
		"exec — a command in the TunnelTug guest. /run, /exec, or {\"type\":\"tool_call\",\"tool\":\"exec\",\"args\":{\"cmd\":\"ls\"}}",
		"search — a lookup. {\"type\":\"tool_call\",\"tool\":\"search\",\"args\":{\"q\":\"the question\"}}",
		"gather — one public page. {\"type\":\"tool_call\",\"tool\":\"gather\",\"args\":{\"url\":\"https://example.com\"}}",
	}, "\n")
}
