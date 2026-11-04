package hodge

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"modelrun/internal/chatfmt"
	"modelrun/internal/tokenizers"
)

func TestSitePrompt(t *testing.T) {
	dir := `C:\Users\grego\EncryptidSystems\stage-ai\models\hodge-1`
	tok, err := tokenizers.Load(dir + `\tokenizer.json`)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(dir + `\tokenizer_config.json`)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	primary, listed := chatfmt.ReadTemplates(cfg["chat_template"])
	has := func(piece string) bool { return tok.ID(piece) >= 0 }
	kind := chatfmt.Choose(has, primary, "")
	if kind != chatfmt.Role {
		t.Fatalf("chat format %s, template %q, listed %d", kind, primary, len(listed))
	}
	nl := tok.Encode("\n\n", false)
	if got := tok.Decode(nl, true); got != "\n\n" {
		t.Fatalf("newline ids %v decode %q", nl, got)
	}
	sys := systemText("Hodge-3.5")
	ids := turnIDs(tok, kind, []int{128000}, sys, "Help me make fried chicken", true)
	got := tok.Decode(ids, true)
	if strings.Contains(got, "\x00") {
		t.Fatalf("nul in %q", got)
	}
	if !strings.Contains(got, "User: Help me make fried chicken<|eot_id|>Assistant:") {
		t.Fatalf("prompt %q", got)
	}
	if !strings.HasPrefix(got, "<|begin_of_text|>System:") {
		t.Fatalf("start %q", got)
	}
}
