package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSkillsList(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MODELRUN_SKILLS", dir)
	if err := os.WriteFile(filepath.Join(dir, "recipe.md"), []byte("# Recipe\n\nWrite the ingredients and the method.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := SkillsText()
	if !strings.Contains(got, "recipe — Write the ingredients and the method.") {
		t.Fatalf("%q", got)
	}
	s, err := Load("recipe")
	if err != nil || !strings.Contains(s.Body, "ingredients") {
		t.Fatalf("%v %#v", err, s)
	}
	if _, err := Load("missing"); err == nil {
		t.Fatal("missing")
	}
}
