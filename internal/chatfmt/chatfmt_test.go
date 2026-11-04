package chatfmt

import "testing"

func has(toks ...string) func(string) bool {
	set := map[string]bool{}
	for _, t := range toks {
		set[t] = true
	}
	return func(s string) bool { return set[s] }
}

func TestChoose(t *testing.T) {
	llama := has("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
	if got := Choose(llama, "", ""); got != Llama3 {
		t.Fatalf("llama with no chat_template: %s", got)
	}
	if got := Choose(llama, roleTemplate, ""); got != Role {
		t.Fatalf("exact 2b4t template stays the role format, got %s", got)
	}
	gemma := has("<start_of_turn>", "<end_of_turn>")
	if got := Choose(gemma, "", "gelu_pytorch_tanh"); got != Gemma {
		t.Fatalf("vaultgemma: %s", got)
	}
	if got := Choose(has("<|im_start|>", "<|im_end|>"), "", ""); got != Qwen {
		t.Fatal("qwen")
	}
	if got := Choose(has("<|start|>", "<|message|>", "<|channel|>"), "", ""); got != Harmony {
		t.Fatal("gpt-oss")
	}
	if got := Choose(has("<|eot_id|>"), "", ""); got != Role {
		t.Fatal("eot without a header is the role format")
	}
	if got := Choose(has(), "", ""); got != Plain {
		t.Fatalf("no markers: %s", got)
	}
	if got := Choose(has(), "", "gelu_pytorch_tanh"); got != Plain {
		t.Fatalf("gemma activation without the turn tokens: %s", got)
	}
}

func TestAvailable(t *testing.T) {
	llama := has("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
	if got := Choose(llama, roleTemplate, ""); got != Role {
		t.Fatalf("default stays role, got %s", got)
	}
	got := Available(llama, roleTemplate, nil, "")
	if len(got) != 2 || got[0] != Role || got[1] != Llama3 {
		t.Fatalf("role template plus llama headers: %v", got)
	}
	primary, listed := ReadTemplates([]any{
		map[string]any{"name": "llama3", "template": "<|start_header_id|>"},
		map[string]any{"name": "role", "template": roleTemplate},
	})
	if primary != roleTemplate || len(listed) != 2 {
		t.Fatalf("list primary %q entries %d", primary, len(listed))
	}
	got = Available(llama, primary, listed, "")
	if k, ok := Pick("llama3", got); !ok || k != Llama3 {
		t.Fatalf("pick llama3 from %v", got)
	}
	if _, ok := Pick("gemma", got); ok {
		t.Fatal("gemma is not available")
	}
	obj, listed := ReadTemplates(map[string]any{"llama3": "h", "role": roleTemplate})
	if obj != roleTemplate || len(listed) != 2 || listed[0].Name != "llama3" {
		t.Fatalf("object %q %#v", obj, listed)
	}
}

func TestRender(t *testing.T) {
	if got := Render(Role, "Be brief.", "Hi", true); got != "System: Be brief.<|eot_id|>User: Hi<|eot_id|>Assistant: " {
		t.Fatalf("role first %q", got)
	}
	if got := Render(Role, "Be brief.", "Next", false); got != "User: Next<|eot_id|>Assistant: " {
		t.Fatalf("role next %q", got)
	}
	wantL := "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	if got := Render(Llama3, "", "Hi", true); got != wantL {
		t.Fatalf("llama %q", got)
	}
	wantG := "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n"
	if got := Render(Gemma, "", "Hi", true); got != wantG {
		t.Fatalf("gemma %q", got)
	}
	wantQ := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
	if got := Render(Qwen, "", "Hi", true); got != wantQ {
		t.Fatalf("qwen %q", got)
	}
	wantH := "<|start|>user<|message|>Hi<|end|><|start|>assistant<|channel|>final<|message|>"
	if got := Render(Harmony, "", "Hi", true); got != wantH {
		t.Fatalf("harmony %q", got)
	}
	if got := Render(Plain, "", "Hi", true); got != "Hi\n" {
		t.Fatalf("plain %q", got)
	}
}
