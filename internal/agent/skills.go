package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Skill is one instruction file.
type Skill struct {
	Name string
	Line string
	Body string
}

// SkillsDir is ~/ModelRun/skills, or MODELRUN_SKILLS.
func SkillsDir() string {
	if v := strings.TrimSpace(os.Getenv("MODELRUN_SKILLS")); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join("ModelRun", "skills")
	}
	return filepath.Join(home, "ModelRun", "skills")
}

// Skills lists the .md files in SkillsDir.
func Skills() ([]Skill, error) {
	entries, err := os.ReadDir(SkillsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Skill
	for _, e := range entries {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".md") {
			continue
		}
		s, err := readSkill(filepath.Join(SkillsDir(), e.Name()))
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// Load reads one skill by name, without the .md.
func Load(name string) (Skill, error) {
	name = strings.TrimSpace(name)
	name = strings.TrimSuffix(name, filepath.Ext(name))
	if name == "" || strings.ContainsAny(name, `/\:`) {
		return Skill{}, fmt.Errorf("no skill %q", name)
	}
	path := filepath.Join(SkillsDir(), name+".md")
	if _, err := os.Stat(path); err != nil {
		return Skill{}, fmt.Errorf("no skill %q", name)
	}
	return readSkill(path)
}

// SkillsText is the /skills reply.
func SkillsText() string {
	all, err := Skills()
	if err != nil {
		return err.Error()
	}
	if len(all) == 0 {
		return "No skills. Add a .md file in " + SkillsDir() + ", then /skill <name>."
	}
	var lines []string
	for _, s := range all {
		if s.Line == "" {
			lines = append(lines, s.Name)
			continue
		}
		lines = append(lines, s.Name+" — "+s.Line)
	}
	return strings.Join(lines, "\n")
}

func readSkill(path string) (Skill, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Skill{}, err
	}
	body := strings.ReplaceAll(string(b), "\r\n", "\n")
	body = strings.TrimSpace(body)
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	if rest, ok := strings.CutPrefix(body, "#"); ok {
		_, after, _ := strings.Cut(rest, "\n")
		body = strings.TrimSpace(after)
	}
	line := ""
	for _, ln := range strings.Split(body, "\n") {
		ln = strings.TrimSpace(ln)
		if ln != "" {
			line = ln
			break
		}
	}
	return Skill{Name: name, Line: line, Body: body}, nil
}
