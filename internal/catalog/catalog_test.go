package catalog

import (
	"os"
	"path/filepath"
	"testing"
)

func TestListHasTernaryAndClock(t *testing.T) {
	all, err := List()
	if err != nil {
		t.Fatal(err)
	}
	var clock bool
	for _, m := range all {
		if m.Family == FamilyHodge {
			t.Fatalf("list invented a checkpoint: %+v", m)
		}
		if m.Name == "dawn/clock" && m.Family == FamilyClock {
			clock = true
		}
	}
	if !clock {
		t.Fatalf("clock missing (%d models)", len(all))
	}
}

func TestLoadReadsTheConfigFile(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"model.bnz", "config.json", "tokenizer.json", "tokenizer_config.json"} {
		body := []byte("x")
		if name == "config.json" {
			body = []byte(`{"hidden_size":8,"num_hidden_layers":2,"vocab_size":32}`)
		}
		if err := os.WriteFile(filepath.Join(dir, name), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cfg := filepath.Join(dir, "run.yaml")
	text := "models:\n  - model: model.bnz\n    tokenizer: tokenizer.json\n    tokenizer_config: tokenizer_config.json\n    config: config.json\n"
	if err := os.WriteFile(cfg, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Load(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if m.Path != filepath.Join(dir, "model.bnz") || m.Config != filepath.Join(dir, "config.json") || m.Hidden != 8 || m.Layers != 2 {
		t.Fatalf("%+v", m)
	}
}
