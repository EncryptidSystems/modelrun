// Command modelrun runs the Hodge-Kuramoto runtime. A packed linear is
// hodgelinear.wgsl on Dawn, limbs included. A dense bf16 linear is dense.wgsl.
// LUT kernels are the WGSL files Dawn loads as written. The config stamp
// model_type is not the runtime name.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"modelrun/internal/agent"
	"modelrun/internal/api"
	"modelrun/internal/catalog"
	"modelrun/internal/generate"
	"modelrun/internal/hodge"
	"modelrun/internal/mark"
	"modelrun/internal/search"
	"modelrun/internal/tui"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		fmt.Print(usage)
		return nil
	}
	switch args[0] {
	case "help", "-h", "--help":
		fmt.Print(usage)
		return nil
	case "list", "ls":
		return list()
	case "ps":
		return ps()
	case "show":
		if len(args) < 2 {
			return fmt.Errorf("modelrun show <model>")
		}
		return show(args[1])
	case "serve":
		addr := "127.0.0.1:11434"
		for i := 1; i < len(args); i++ {
			if args[i] == "-addr" && i+1 < len(args) {
				addr = args[i+1]
				i++
			}
		}
		hodge.SetLog(func(s string) { fmt.Fprintln(os.Stderr, s) })
		fmt.Println("modelrun serve", addr)
		return http.ListenAndServe(addr, api.Handler())
	case "run":
		name, prompt, config, opt, err := parseRun(args[1:])
		if err != nil {
			return err
		}
		if prompt == "" && isTTY() {
			if !opt.TempSet && !opt.TopSet {
				opt.ConfigSample = true
			}
			if config != "" {
				return tui.RunFile(config, opt)
			}
			return tui.Run(name, opt)
		}
		if prompt == "" {
			b, err := io.ReadAll(os.Stdin)
			if err != nil || len(strings.TrimSpace(string(b))) == 0 {
				return fmt.Errorf("modelrun run %s: prompt missing", name)
			}
			prompt = string(b)
		}
		if text, ok := verbText(prompt); ok {
			fmt.Println(text)
			return nil
		}
		user := prompt
		var sources string
		target := name
		if config != "" {
			target = config
		}
		hodge.ClearInterrupt()
		route := agent.Talk
		if decided, terr := generate.Think(target, prompt); terr != nil {
			if errors.Is(terr, hodge.ErrInterrupted) {
				return terr
			}
		} else {
			route = decided
		}
		if route == agent.Fact || route == agent.Recipe {
			if route == agent.Fact {
				var gerr error
				user, sources, gerr = search.New().Ground(context.Background(), prompt, func(s string) {
					fmt.Fprintln(os.Stderr, s)
				}, func(label, body string) {
					fmt.Fprintf(os.Stderr, "%s\n%s\n\n", label, body)
				})
				if gerr != nil {
					return gerr
				}
			}
			hodge.SetLog(func(s string) { fmt.Fprintln(os.Stderr, s) })
			text, serr := generate.Shown(target, prompt, route, user, sources, opt)
			if serr != nil {
				return serr
			}
			if strings.TrimSpace(text) != "" {
				fmt.Println(text)
			}
			reportMark(name, config, text)
			return nil
		}
		hodge.SetLog(func(s string) { fmt.Fprintln(os.Stderr, s) })
		var rep generate.Reply
		if config != "" {
			rep, err = generate.ChatFile(config, []generate.Turn{{Role: "user", Content: user}}, opt, func(s string) {
				fmt.Fprint(os.Stdout, s)
			})
		} else {
			rep, err = generate.Chat(name, []generate.Turn{{Role: "user", Content: user}}, opt, func(s string) {
				fmt.Fprint(os.Stdout, s)
			})
		}
		if err != nil {
			return err
		}
		if search.OffSubject(prompt, rep.Text) {
			var gerr error
			user, sources, gerr = search.New().Ground(context.Background(), prompt, func(s string) {
				fmt.Fprintln(os.Stderr, s)
			}, func(label, body string) {
				fmt.Fprintf(os.Stderr, "%s\n%s\n\n", label, body)
			})
			if gerr != nil {
				return gerr
			}
			text, serr := generate.Shown(target, prompt, agent.Fact, user, sources, opt)
			if serr != nil {
				return serr
			}
			if strings.TrimSpace(text) != "" {
				fmt.Println(text)
			}
			reportMark(name, config, text)
			return nil
		}
		if rep.Text == "" {
			fmt.Println(rep.Text)
		} else if !strings.HasSuffix(rep.Text, "\n") {
			fmt.Println()
		}
		if strings.TrimSpace(sources) != "" {
			fmt.Println()
			fmt.Println(strings.TrimSpace(sources))
		}
		reportMark(name, config, rep.Text)
		return nil
	default:
		return fmt.Errorf("unknown %q\n%s", args[0], usage)
	}
}

func verbText(prompt string) (string, bool) {
	name, arg, ok := agent.Verb(prompt)
	if !ok {
		return "", false
	}
	switch name {
	case "tools":
		return agent.ToolsText(), true
	case "skills":
		return agent.SkillsText(), true
	case "skill":
		if strings.TrimSpace(arg) == "" {
			hodge.UseSkill("")
			return "Skill off.", true
		}
		s, err := agent.Load(arg)
		if err != nil {
			return err.Error(), true
		}
		hodge.UseSkill(s.Body)
		return "Skill " + s.Name + ".\n\n" + s.Body, true
	default:
		return "", false
	}
}

func parseRun(args []string) (name, prompt, config string, opt generate.Opt, err error) {
	opt = generate.Opt{ConfigSample: true}
	i := 0
	for i < len(args) {
		a := args[i]
		if a == "--" {
			i++
			break
		}
		if !strings.HasPrefix(a, "-") || a == "-" {
			break
		}
		switch a {
		case "-n":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("modelrun run: -n needs a number")
			}
			n, e := strconv.Atoi(args[i+1])
			if e != nil || n < 1 {
				return "", "", "", opt, fmt.Errorf("modelrun run: -n %q", args[i+1])
			}
			opt.N = n
			i += 2
		case "-config", "--config":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("modelrun run: --config needs a file")
			}
			config = args[i+1]
			i += 2
		case "-temp":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("modelrun run: -temp needs a number")
			}
			t, e := strconv.ParseFloat(args[i+1], 64)
			if e != nil || t < 0 {
				return "", "", "", opt, fmt.Errorf("modelrun run: -temp %q", args[i+1])
			}
			opt.Temp = t
			opt.TempSet = true
			i += 2
		case "-top-p":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("modelrun run: -top-p needs a number")
			}
			p, e := strconv.ParseFloat(args[i+1], 64)
			if e != nil || p < 0 {
				return "", "", "", opt, fmt.Errorf("modelrun run: -top-p %q", args[i+1])
			}
			opt.TopP = p
			opt.TopSet = true
			i += 2
		default:
			return "", "", "", opt, fmt.Errorf("unknown flag %q\n%s", a, usage)
		}
	}
	if config != "" {
		prompt = strings.Join(args[i:], " ")
		return "", prompt, config, opt, nil
	}
	if i >= len(args) {
		return "", "", "", opt, fmt.Errorf("modelrun run --config FILE [prompt]")
	}
	name = args[i]
	prompt = strings.Join(args[i+1:], " ")
	return name, prompt, "", opt, nil
}

func list() error {
	all, err := catalog.List()
	if err != nil {
		return err
	}
	for _, m := range all {
		extra := m.Detail
		if m.Family == catalog.FamilyHodge && m.Layers > 0 {
			extra = fmt.Sprintf("%s  %s  %d layers  hidden %d", m.Detail, m.Kind, m.Layers, m.Hidden)
		}
		fmt.Printf("%-28s %-8s %s\n", m.Name, m.Family, extra)
	}
	return nil
}

func show(name string) error {
	m, ok, err := catalog.Get(name)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no model %q", name)
	}
	fmt.Printf("name     %s\nfamily   %s\nkind     %s\n", m.Name, m.Family, m.Kind)
	if m.Detail != "" {
		fmt.Printf("detail   %s\n", m.Detail)
	}
	if m.Layers > 0 {
		fmt.Printf("layers   %d\nhidden   %d\nvocab    %d\n", m.Layers, m.Hidden, m.Vocab)
	}
	if m.Bytes > 0 {
		fmt.Printf("bytes    %d\n", m.Bytes)
	}
	if m.Path != "" {
		fmt.Printf("path     %s\n", m.Path)
	}
	return nil
}

func reportMark(name, config, text string) {
	var tok, cfg string
	if config != "" {
		if m, err := catalog.Load(config); err == nil {
			tok, cfg = m.Tokenizer, m.TokenizerConfig
		}
	} else if m, err := catalog.Load(name); err == nil {
		tok, cfg = m.Tokenizer, m.TokenizerConfig
	} else if m, ok, err := catalog.Get(name); err == nil && ok {
		tok, cfg = m.Tokenizer, m.TokenizerConfig
	}
	c, err := mark.Cached(tok, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "watermark:", err)
		return
	}
	if c == nil || strings.TrimSpace(text) == "" {
		return
	}
	s := c.Score(text)
	if s.Words == 0 {
		return
	}
	kind := "clear"
	if s.Marked {
		kind = "marked"
	} else if s.Words < 24 {
		kind = "short"
	}
	fmt.Fprintf(os.Stderr, "watermark  %s  words %d  alignment %.3f  null %.3f  cell %.4f  cell_null %.4f\n", kind, s.Words, s.Alignment, s.NullMax, s.CellShare, s.CellNull)
}

func ps() error {
	got := generate.Loaded()
	if len(got) == 0 {
		fmt.Println("no models loaded")
		return nil
	}
	for name, at := range got {
		fmt.Printf("%s  %s\n", name, at.Format("15:04:05"))
	}
	return nil
}

func isTTY() bool {
	fi, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

const usage = `modelrun runs the Hodge-Kuramoto runtime. A packed linear is hodgelinear.wgsl
on Dawn, limbs included. A dense bf16 linear is dense.wgsl. LUT kernels are the
WGSL files Dawn loads as written. The config stamp model_type is not the runtime name.

  modelrun list
  modelrun run [-n N] [-temp T] [-top-p P] --config FILE [text]
  modelrun show <model>
  modelrun ps
  modelrun serve [-addr 127.0.0.1:11434]

Omit -n to run until the model stops. -n N caps the answer.
Omit -temp to use the checkpoint sampler. -temp 0 is greedy.
The HTTP routes match Ollama: /api/tags /api/chat /api/generate /api/show /api/ps.
When model.cochain.safetensors sits beside the tokenizer, each reply is scored
against the vocabulary as it is produced. The HTTP object carries watermark.
modelrun run prints the same line on stderr.
`
