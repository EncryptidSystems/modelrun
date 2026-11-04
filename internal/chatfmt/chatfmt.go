// Package chatfmt is the chat turn of each model Champ runs. The format comes from
// that model's own tokens and chat_template, not from one template shared by every run.
package chatfmt

import (
	"sort"
	"strings"
)

// Kind is one model's chat format.
type Kind string

const (
	// Role is the 2B4T base's format: "Role: content<|eot_id|>", then "Assistant: ".
	Role Kind = "role"
	// Llama3 is Llama 3's header turns.
	Llama3 Kind = "llama3"
	// Gemma is VaultGemma and Gemma: start_of_turn / end_of_turn, model as the assistant.
	Gemma Kind = "gemma"
	// Qwen is the im_start / im_end turn, including a text Qwen packed for Champ.
	Qwen Kind = "qwen"
	// Harmony is gpt-oss: start / channel / message, answering on the final channel.
	Harmony Kind = "harmony"
	// Plain is a checkpoint with none of those markers. The user text is the prompt.
	Plain Kind = "plain"
)

// The 2B4T tokenizer_config chat_template, matched whole.
const roleTemplate = `{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] | capitalize + ': '+ message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}`

// Listed is one named chat_template. A string chat_template has no list.
type Listed struct {
	Name string
	Body string
}

// ReadTemplates accepts chat_template as a string, a list of {name, template},
// or an object of name to template. primary is the body Choose uses. When the
// role template is one of several, it stays primary.
func ReadTemplates(v any) (primary string, listed []Listed) {
	switch t := v.(type) {
	case string:
		return t, nil
	case []any:
		for _, item := range t {
			o, _ := item.(map[string]any)
			if o == nil {
				continue
			}
			name, _ := o["name"].(string)
			body, _ := o["template"].(string)
			if body == "" {
				body, _ = o["chat_template"].(string)
			}
			if body == "" {
				continue
			}
			listed = append(listed, Listed{Name: name, Body: body})
		}
	case map[string]any:
		names := make([]string, 0, len(t))
		for k := range t {
			names = append(names, k)
		}
		sort.Strings(names)
		for _, name := range names {
			body, _ := t[name].(string)
			if body == "" {
				continue
			}
			listed = append(listed, Listed{Name: name, Body: body})
		}
	}
	for _, l := range listed {
		if l.Body == roleTemplate {
			return l.Body, listed
		}
	}
	if len(listed) > 0 {
		return listed[0].Body, listed
	}
	return "", listed
}

// ParseKind is a format name from a template entry or a -format flag.
func ParseKind(name string) (Kind, bool) {
	switch Kind(name) {
	case Role, Llama3, Gemma, Qwen, Harmony, Plain:
		return Kind(name), true
	default:
		return "", false
	}
}

func tokensOK(has func(string) bool, k Kind) bool {
	switch k {
	case Role:
		return has("<|eot_id|>")
	case Llama3:
		return has("<|start_header_id|>") && has("<|end_header_id|>") && has("<|eot_id|>")
	case Gemma:
		return has("<start_of_turn>") && has("<end_of_turn>")
	case Qwen:
		return has("<|im_start|>") && has("<|im_end|>")
	case Harmony:
		return has("<|start|>") && has("<|message|>") && has("<|channel|>")
	case Plain:
		return true
	default:
		return false
	}
}

// Choose is the default format. has reports whether the tokenizer has a piece.
// hiddenActivation is config.json's hidden_activation (set on Gemma, including VaultGemma).
// An exact 2B4T template wins, so a file that also carries other specials stays
// the role format until the run asks for another one from Available.
func Choose(has func(string) bool, template, hiddenActivation string) Kind {
	if has == nil {
		has = func(string) bool { return false }
	}
	if template == roleTemplate {
		return Role
	}
	if has("<|start_header_id|>") && has("<|end_header_id|>") && has("<|eot_id|>") {
		return Llama3
	}
	// Gemma's turn tokens, including VaultGemma. hiddenActivation alone is not enough:
	// the markers have to be pieces, or the encoder would split them into bytes.
	if has("<start_of_turn>") && has("<end_of_turn>") {
		return Gemma
	}
	if has("<|im_start|>") && has("<|im_end|>") {
		return Qwen
	}
	if has("<|start|>") && has("<|message|>") && has("<|channel|>") {
		return Harmony
	}
	if has("<|eot_id|>") {
		return Role
	}
	return Plain
}

// Available is every format this checkpoint can render. The default from Choose
// is first. A role template does not hide Llama 3 when the header tokens are pieces.
func Available(has func(string) bool, template string, listed []Listed, hiddenActivation string) []Kind {
	if has == nil {
		has = func(string) bool { return false }
	}
	var out []Kind
	add := func(k Kind) {
		for _, x := range out {
			if x == k {
				return
			}
		}
		out = append(out, k)
	}
	add(Choose(has, template, hiddenActivation))
	if template == roleTemplate || tokensOK(has, Role) {
		add(Role)
	}
	order := []Kind{Llama3, Gemma, Qwen, Harmony}
	for _, k := range order {
		if tokensOK(has, k) {
			add(k)
		}
	}
	for _, l := range listed {
		k, ok := ParseKind(l.Name)
		if ok && tokensOK(has, k) {
			add(k)
		}
		if l.Body == roleTemplate {
			add(Role)
		}
	}
	return out
}

// Pick is name among have. An empty name is the default, which is have[0].
func Pick(name string, have []Kind) (Kind, bool) {
	if name == "" {
		if len(have) == 0 {
			return Plain, false
		}
		return have[0], true
	}
	if k, ok := ParseKind(name); ok {
		for _, x := range have {
			if x == k {
				return k, true
			}
		}
	}
	return "", false
}

// Name is the format's label. It is a string literal so the wasm compiler can emit it.
func Name(k Kind) string {
	switch k {
	case Role:
		return "role"
	case Llama3:
		return "llama3"
	case Gemma:
		return "gemma"
	case Qwen:
		return "qwen"
	case Harmony:
		return "harmony"
	default:
		return "plain"
	}
}

// Stops are the pieces that end an answer, in the order a carried turn token prefers.
func Stops(k Kind) []string {
	switch k {
	case Role, Llama3:
		return []string{"<|eot_id|>", "<|eom_id|>"}
	case Gemma:
		return []string{"<end_of_turn>", "<eos>"}
	case Qwen:
		return []string{"<|im_end|>"}
	case Harmony:
		return []string{"<|return|>", "<|end|>", "<|call|>"}
	default:
		return nil
	}
}

// Render is the text of one user turn, including the assistant prefix the model continues.
// first is the start of the chat, where system is applied. Later turns are the user message only.
func Render(k Kind, system, user string, first bool) string {
	user = strings.TrimSpace(user)
	system = strings.TrimSpace(system)
	switch k {
	case Llama3:
		var b strings.Builder
		if first && system != "" {
			writeLlama(&b, "system", system)
		}
		writeLlama(&b, "user", user)
		b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
		return b.String()
	case Gemma:
		var b strings.Builder
		b.WriteString("<start_of_turn>user\n")
		if first && system != "" {
			b.WriteString(system)
			b.WriteString("\n\n")
		}
		b.WriteString(user)
		b.WriteString("<end_of_turn>\n<start_of_turn>model\n")
		return b.String()
	case Qwen:
		var b strings.Builder
		if first && system != "" {
			writeQwen(&b, "system", system)
		}
		writeQwen(&b, "user", user)
		b.WriteString("<|im_start|>assistant\n")
		return b.String()
	case Harmony:
		var b strings.Builder
		if first && system != "" {
			b.WriteString("<|start|>system<|message|>")
			b.WriteString(system)
			b.WriteString("<|end|>")
		}
		b.WriteString("<|start|>user<|message|>")
		b.WriteString(user)
		b.WriteString("<|end|><|start|>assistant<|channel|>final<|message|>")
		return b.String()
	case Plain:
		if first && system != "" {
			return system + "\n\n" + user + "\n"
		}
		return user + "\n"
	default:
		var b strings.Builder
		if first && system != "" {
			b.WriteString(roleLine("system", system))
		}
		b.WriteString(roleLine("user", user))
		b.WriteString("Assistant: ")
		return b.String()
	}
}

func writeLlama(b *strings.Builder, role, content string) {
	b.WriteString("<|start_header_id|>")
	b.WriteString(role)
	b.WriteString("<|end_header_id|>\n\n")
	b.WriteString(content)
	b.WriteString("<|eot_id|>")
}

func writeQwen(b *strings.Builder, role, content string) {
	b.WriteString("<|im_start|>")
	b.WriteString(role)
	b.WriteString("\n")
	b.WriteString(content)
	b.WriteString("<|im_end|>\n")
}

func roleLine(role, content string) string {
	return capitalize(role) + ": " + content + "<|eot_id|>"
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + strings.ToLower(s[1:])
}
