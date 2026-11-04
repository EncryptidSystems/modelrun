package tui

import (
	"strings"
	"testing"

	"modelrun/internal/catalog"
	"modelrun/internal/generate"
)

func TestInputAtBottom(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge}, generate.Opt{})
	m.loaded = true
	m.width = 80
	m.height = 24
	view := m.View()
	lines := strings.Split(view, "\n")
	if len(lines) < 20 {
		t.Fatalf("view has %d lines\n%s", len(lines), view)
	}
	top := strings.Join(lines[:8], "\n")
	bottom := strings.Join(lines[len(lines)-8:], "\n")
	if strings.Contains(top, "╰") || strings.Contains(top, "┘") {
		t.Fatalf("box starts at the top\n%s", view)
	}
	if !strings.Contains(bottom, "╰") && !strings.Contains(bottom, "╯") && !strings.Contains(bottom, "┘") {
		t.Fatalf("box is not at the bottom\n%s", view)
	}
}
