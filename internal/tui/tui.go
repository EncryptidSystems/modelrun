// Package tui is the Bubble Tea face of ModelRun.
package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"modelrun/internal/agent"
	"modelrun/internal/catalog"
	"modelrun/internal/generate"
	"modelrun/internal/hodge"
	"modelrun/internal/search"
	"modelrun/internal/vm"
)

var (
	titleStyle = lipgloss.NewStyle().Bold(true)
	dimStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	accent     = lipgloss.NewStyle().Foreground(lipgloss.Color("213"))
	bad        = lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	youStyle   = lipgloss.NewStyle().Bold(true)
	boxStyle   = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("238")).
			Padding(0, 1)
)

type turn struct {
	role  string
	text  string
	label string
}

type replyMsg struct {
	text string
	err  error
}

type tokenMsg struct {
	text string
	done bool
	err  error
}

type statusMsg struct{ text string }

type progressMsg struct{ text string }

type loadedMsg struct{ err error }

type infoMsg struct{ text string }

type toolMsg struct {
	label string
	text  string
}

type replaceMsg struct{ text string }

type model struct {
	cur      catalog.Model
	turns    []turn
	skill    string
	input    textarea.Model
	spin     spinner.Model
	busy     bool
	width    int
	height   int
	status   string
	errText  string
	opt      generate.Opt
	stream   chan tea.Msg
	scroll   int
	loading  bool
	loaded   bool
	progress float64
	escArmed bool
}

// RunFile starts chat for the checkpoint named by a config file.
func RunFile(path string, opt generate.Opt) error {
	meta, err := catalog.Load(path)
	if err != nil {
		return err
	}
	m := newModel(meta, opt)
	m.loading = true
	m.input.Focus()
	_, err = tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion()).Run()
	return err
}

// Run starts chat for one checkpoint. The name comes from the CLI.
func Run(name string, opt generate.Opt) error {
	if name == "" {
		return fmt.Errorf("modelrun run <model>")
	}
	meta, ok, err := catalog.Get(name)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no model %q (modelrun list)", name)
	}
	m := newModel(meta, opt)
	if meta.Family == catalog.FamilyHodge {
		m.loading = true
	} else {
		m.loaded = true
	}
	m.input.Focus()
	_, err = tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion()).Run()
	return err
}

func newModel(cur catalog.Model, opt generate.Opt) model {
	ta := textarea.New()
	ta.Placeholder = "Message"
	ta.CharLimit = 0
	ta.SetHeight(3)
	ta.ShowLineNumbers = false
	ta.Prompt = ""
	ta.KeyMap.InsertNewline.SetKeys("shift+enter", "alt+enter")
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = accent
	return model{
		cur:    cur,
		input:  ta,
		spin:   sp,
		width:  80,
		height: 24,
		opt:    opt,
	}
}

func (m model) Init() tea.Cmd {
	cmds := []tea.Cmd{textarea.Blink, m.spin.Tick}
	if m.loading {
		cmds = append(cmds, loadCmd(m.cur.Name))
	}
	return tea.Batch(cmds...)
}

func (m model) current() (catalog.Model, bool) {
	if m.cur.Name == "" {
		return catalog.Model{}, false
	}
	return m.cur, true
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.input.SetWidth(msg.Width - 4)
		return m, nil
	case infoMsg:
		m.status = msg.text
		return m, nil
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd
	case replyMsg:
		m.busy = false
		if msg.err != nil {
			m.turns = append(m.turns, turn{"error", msg.err.Error(), ""})
		} else {
			m.turns = append(m.turns, turn{"assistant", msg.text, ""})
		}
		return m, nil
	case chan tea.Msg:
		m.stream = msg
		return m, waitStream(msg)
	case progressMsg:
		if f := progressOf(msg.text); f >= 0 {
			m.progress = f
		}
		m.status = msg.text
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case loadedMsg:
		m.loading = false
		m.stream = nil
		if msg.err != nil {
			m.turns = append(m.turns, turn{"error", msg.err.Error(), ""})
			m.status = msg.err.Error()
			return m, nil
		}
		m.loaded = true
		m.progress = 1
		m.status = "ready"
		return m, nil
	case statusMsg:
		m.status = msg.text
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case tokenMsg:
		if msg.done {
			m.busy = false
			m.stream = nil
			if errors.Is(msg.err, hodge.ErrInterrupted) {
				m.status = "interrupted"
			} else if msg.err != nil {
				m.turns = append(m.turns, turn{"error", msg.err.Error(), ""})
			}
			return m, nil
		}
		m = m.appendPiece(msg.text)
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case tea.MouseMsg:
		m.escArmed = false
		switch msg.Button {
		case tea.MouseButtonWheelUp:
			m.scroll += 3
		case tea.MouseButtonWheelDown:
			if m.scroll > 3 {
				m.scroll -= 3
			} else {
				m.scroll = 0
			}
		}
		return m, nil
	case replaceMsg:
		if n := len(m.turns); n > 0 && m.turns[n-1].role == "assistant" {
			if strings.TrimSpace(msg.text) == "" {
				m.turns = m.turns[:n-1]
			} else {
				m.turns[n-1].text = msg.text
			}
		} else if strings.TrimSpace(msg.text) != "" {
			m.turns = append(m.turns, turn{role: "assistant", text: msg.text})
		}
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case toolMsg:
		label := msg.label
		if label == "" {
			label = "sandbox"
		}
		next := turn{role: "tool", label: label, text: msg.text}
		if n := len(m.turns); n > 0 && m.turns[n-1].role == "assistant" && agent.Parse(m.turns[n-1].text) != nil {
			m.turns[n-1] = next
		} else {
			m.turns = append(m.turns, next)
		}
		m.scroll = 0
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case tea.KeyMsg:
		if msg.String() != "esc" {
			m.escArmed = false
		}
		switch msg.String() {
		case "ctrl+c":
			if m.busy {
				hodge.Interrupt()
				agent.Cancel()
				m.status = "interrupted"
			}
			return m, nil
		case "esc":
			if m.escArmed {
				return m, tea.Quit
			}
			m.escArmed = true
			m.status = "esc again to quit"
			return m, nil
		case "pgup", "ctrl+u":
			m.scroll += m.page()
			return m, nil
		case "pgdown", "ctrl+d":
			if m.scroll > 0 {
				m.scroll -= m.page()
				if m.scroll < 0 {
					m.scroll = 0
				}
				return m, nil
			}
		case "enter":
			if m.busy || m.loading || !m.loaded {
				return m, nil
			}
			text := strings.TrimSpace(m.input.Value())
			if text == "" {
				return m, nil
			}
			cur, ok := m.current()
			if !ok {
				return m, nil
			}
			m.input.Reset()
			m.turns = append(m.turns, turn{"user", text, ""})
			m.scroll = 0
			if name, arg, ok := agent.Verb(text); ok {
				var reply string
				var err error
				m, reply, err = m.verb(name, arg)
				if err != nil {
					m.turns = append(m.turns, turn{"error", err.Error(), ""})
				} else if reply != "" {
					m.turns = append(m.turns, turn{"assistant", reply, ""})
				}
				return m, nil
			}
			m.busy = true
			if call := agent.Slash(text); call != nil {
				m.status = "sandbox"
				return m, execCmd(call.Cmd())
			}
			if cur.Family == catalog.FamilyHodge {
				return m, streamCmd(cur.Name, text, m.opt)
			}
			return m, func() tea.Msg {
				rep, err := generate.Run(cur.Name, text)
				if err != nil {
					return replyMsg{err: err}
				}
				return replyMsg{text: rep.Text}
			}
		}
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m model) appendPiece(s string) model {
	if s == "" {
		return m
	}
	if n := len(m.turns); n > 0 && m.turns[n-1].role == "assistant" {
		m.turns[n-1].text += s
		return m
	}
	m.turns = append(m.turns, turn{"assistant", s, ""})
	return m
}

func loadCmd(name string) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg, 64)
		go func() {
			defer close(ch)
			hodge.SetLog(func(s string) {
				ch <- progressMsg{text: s}
			})
			defer hodge.SetLog(nil)
			err := hodge.Prepare(name)
			ch <- loadedMsg{err: err}
		}()
		return ch
	}
}

func progressOf(s string) float64 {
	switch {
	case strings.HasPrefix(s, "decoding"):
		return 0.08
	case strings.HasPrefix(s, "decoded"):
		return 0.28
	case strings.HasPrefix(s, "uploading"):
		return 0.34
	case strings.HasPrefix(s, "weights"):
		return 0.97
	case strings.HasPrefix(s, "ready"):
		return 1
	}
	var n int
	if _, err := fmt.Sscanf(s, "layer %d", &n); err == nil {
		if n < 0 {
			n = 0
		}
		if n > 29 {
			n = 29
		}
		return 0.36 + 0.60*float64(n+1)/30
	}
	return -1
}

func streamCmd(name, text string, opt generate.Opt) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg, 32)
		go func() {
			defer close(ch)
			hodge.ClearInterrupt()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			agent.Arm(cancel)
			defer agent.Arm(nil)
			hodge.SetLog(func(s string) {
				select {
				case ch <- statusMsg{text: s}:
				default:
				}
			})
			defer hodge.SetLog(nil)
			ask := func(user string) (string, error) {
				var b strings.Builder
				_, err := hodge.Continue(name, user, opt, func(s string) {
					b.WriteString(s)
					ch <- tokenMsg{text: s}
				})
				return b.String(), err
			}
			quiet := func(user string) (string, error) {
				return hodge.Continue(name, user, opt, nil)
			}
			ch <- statusMsg{text: "thinking…"}
			route := agent.Talk
			if decided, terr := generate.Think(name, text); terr != nil {
				if hodge.Interrupted() || ctx.Err() != nil {
					ch <- tokenMsg{done: true, err: hodge.ErrInterrupted}
					return
				}
			} else {
				route = decided
			}
			var reply string
			var err error
			switch route {
			case agent.Fact, agent.Recipe:
				var prompt, sources string
				if route == agent.Fact {
					web := search.New()
					var gerr error
					prompt, sources, gerr = web.Ground(ctx, text, func(s string) {
						ch <- statusMsg{text: s}
					}, func(label, body string) {
						ch <- toolMsg{label: label, text: body}
					})
					if hodge.Interrupted() || ctx.Err() != nil {
						ch <- tokenMsg{done: true, err: hodge.ErrInterrupted}
						return
					}
					if gerr != nil && prompt == "" {
						ch <- tokenMsg{done: true, err: gerr}
						return
					}
				}
				reply, err = generate.Shown(name, text, route, prompt, sources, opt)
				if err == nil && !hodge.Interrupted() && strings.TrimSpace(reply) != "" {
					ch <- tokenMsg{text: reply}
				}
			case agent.Command:
				reply, err = quiet(text)
				box := vm.New()
				for round := 0; round < agent.MaxRounds && err == nil && !hodge.Interrupted(); round++ {
					call := agent.Alone(reply)
					if call == nil {
						break
					}
					ch <- statusMsg{text: call.Tool + "…"}
					label, out, ok := runTool(ctx, box, call)
					if !ok {
						break
					}
					if hodge.Interrupted() || ctx.Err() != nil {
						ch <- toolMsg{label: label, text: "Interrupted."}
						err = hodge.ErrInterrupted
						break
					}
					ch <- toolMsg{label: label, text: out}
					next := agent.Wire(out)
					if call.Tool != "exec" {
						next = agent.LookupWire(out)
					}
					reply, err = quiet(next)
				}
				if err == nil && !hodge.Interrupted() && strings.TrimSpace(reply) != "" {
					if agent.Alone(reply) == nil {
						ch <- tokenMsg{text: reply}
					} else {
						reply, err = ask("Answer the person in plain words. Do not use JSON.")
					}
				}
			default:
				reply, err = ask(text)
				if err == nil && !hodge.Interrupted() && search.OffSubject(text, reply) {
					web := search.New()
					prompt, sources, gerr := web.Ground(ctx, text, func(s string) {
						ch <- statusMsg{text: s}
					}, func(label, body string) {
						ch <- toolMsg{label: label, text: body}
					})
					if gerr != nil && prompt == "" {
						ch <- replaceMsg{}
					} else if !hodge.Interrupted() && ctx.Err() == nil {
						shown, serr := generate.Shown(name, text, agent.Fact, prompt, sources, opt)
						if serr != nil {
							err = serr
						} else {
							ch <- replaceMsg{text: shown}
						}
					}
				}
			}
			ch <- tokenMsg{done: true, err: err}
		}()
		return ch
	}
}

func execCmd(command string) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg, 4)
		go func() {
			defer close(ch)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			agent.Arm(cancel)
			defer agent.Arm(nil)
			hodge.ClearInterrupt()
			out, _ := vm.New().Exec(ctx, command)
			if hodge.Interrupted() || ctx.Err() != nil {
				ch <- toolMsg{text: "$ " + command + "\nInterrupted."}
				ch <- tokenMsg{done: true, err: hodge.ErrInterrupted}
				return
			}
			ch <- toolMsg{text: "$ " + command + "\n" + out}
			ch <- tokenMsg{done: true}
		}()
		return ch
	}
}

func waitStream(ch chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return tokenMsg{done: true}
		}
		return msg
	}
}

func (m model) View() string {
	width := m.width
	if width < 20 {
		width = 20
	}
	left := titleStyle.Render("modelrun")
	right := dimStyle.Render(m.modelLabel())
	gap := width - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 1 {
		gap = 1
	}
	head := left + strings.Repeat(" ", gap) + right
	rule := dimStyle.Render(strings.Repeat("─", width))
	foot := m.footer()
	footH := strings.Count(foot, "\n") + 1
	room := m.height - 2 - footH
	if room < 8 {
		room = 8
	}
	return head + "\n" + rule + "\n" + m.viewChat(room) + "\n" + foot
}

func (m model) modelLabel() string {
	cur, ok := m.current()
	if !ok {
		return ""
	}
	if cur.Detail != "" {
		return cur.Detail
	}
	return cur.Name
}

func (m model) footer() string {
	width := m.width
	if width < 20 {
		width = 20
	}
	state := "ready"
	if m.skill != "" {
		state = "skill " + m.skill
	}
	if m.busy {
		state = m.spin.View() + " " + oneLine(m.status)
		if strings.TrimSpace(m.status) == "" {
			state = m.spin.View() + " working"
		}
	}
	keys := "enter send   /tools   /skills   /skill name   esc esc quit"
	status := dimStyle.Render(clip(m.modelLabel()+"  ·  "+state, width))
	hint := dimStyle.Render(clip(keys, width))
	return status + "\n" + hint
}

func (m model) viewChat(room int) string {
	width := m.width
	if width < 20 {
		width = 20
	}
	name := m.modelLabel()
	if name == "" {
		name = "Hodge"
	}
	if m.loading || !m.loaded {
		return m.loadingView(width, name)
	}
	m.input.SetWidth(width - 6)
	box := boxStyle.Width(width - 4).Render(m.input.View())
	boxH := strings.Count(box, "\n") + 1
	viewH := room - boxH - 1
	if viewH < 1 {
		viewH = 1
	}
	lines := m.transcript(width)
	if m.scroll > len(lines)-viewH {
		m.scroll = max(0, len(lines)-viewH)
	}
	end := len(lines) - m.scroll
	start := end - viewH
	if start < 0 {
		start = 0
	}
	if end < start {
		end = start
	}
	if end > len(lines) {
		end = len(lines)
	}
	var shown []string
	if start < end {
		shown = lines[start:end]
	}
	for len(shown) < viewH {
		shown = append(shown, "")
	}
	return strings.Join(shown, "\n") + "\n\n" + box
}

func (m model) loadingView(width int, name string) string {
	inner := width - 2
	if inner < 8 {
		inner = 8
	}
	filled := int(m.progress * float64(inner))
	if filled < 0 {
		filled = 0
	}
	if filled > inner {
		filled = inner
	}
	bar := accent.Render(strings.Repeat("█", filled)) + dimStyle.Render(strings.Repeat("░", inner-filled))
	pct := fmt.Sprintf("%d%%", int(m.progress*100+0.5))
	label := oneLine(m.status)
	if label == "" {
		label = "loading"
	}
	return "\n" + titleStyle.Render("Loading "+name) + "\n\n" + bar + "  " + dimStyle.Render(pct) + "\n\n" + dimStyle.Render(clip(label, width))
}

func (m model) transcript(width int) []string {
	wrapAt := width - 2
	if wrapAt < 8 {
		wrapAt = 8
	}
	name := m.modelLabel()
	if name == "" {
		name = "Hodge"
	}
	var lines []string
	for _, t := range m.turns {
		switch t.role {
		case "user":
			lines = append(lines, youStyle.Render("You"))
			lines = append(lines, wrapText(t.text, wrapAt)...)
		case "error":
			lines = append(lines, bad.Render(t.text))
		case "tool":
			label := t.label
			if label == "" {
				label = "sandbox"
			}
			lines = append(lines, dimStyle.Render(label))
			lines = append(lines, wrapText(t.text, wrapAt)...)
		default:
			lines = append(lines, accent.Render(name))
			lines = append(lines, wrapText(t.text, wrapAt)...)
		}
		lines = append(lines, "")
	}
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func (m model) page() int {
	n := m.height / 2
	if n < 1 {
		n = 1
	}
	return n
}

func (m model) verb(name, arg string) (model, string, error) {
	switch name {
	case "tools":
		return m, agent.ToolsText(), nil
	case "skills":
		return m, agent.SkillsText(), nil
	case "skill":
		if strings.TrimSpace(arg) == "" {
			m.skill = ""
			hodge.UseSkill("")
			return m, "Skill off.", nil
		}
		s, err := agent.Load(arg)
		if err != nil {
			return m, "", err
		}
		m.skill = s.Name
		hodge.UseSkill(s.Body)
		return m, "Skill " + s.Name + ".", nil
	default:
		return m, "", nil
	}
}

func runTool(ctx context.Context, box *vm.Client, call *agent.Call) (label, out string, ok bool) {
	switch call.Tool {
	case "exec":
		if call.Cmd() == "" {
			return "sandbox", "Missing command.", true
		}
		text, _ := box.Exec(ctx, call.Cmd())
		return "sandbox", "$ " + call.Cmd() + "\n" + text, true
	case "search":
		text, err := search.New().Search(ctx, call.Query(), call.Source())
		if text == "" && err != nil {
			text = "The lookup failed."
		}
		return "search", text, true
	case "gather":
		text, err := search.New().Gather(ctx, call.URL())
		if text == "" && err != nil {
			text = "The lookup failed."
		}
		return "gather", text, true
	default:
		return "", "", false
	}
}

func wrapText(s string, width int) []string {
	if width < 8 {
		width = 8
	}
	var out []string
	for _, para := range strings.Split(s, "\n") {
		out = append(out, wrapLine(para, width)...)
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func wrapLine(s string, width int) []string {
	r := []rune(s)
	if len(r) == 0 {
		return []string{""}
	}
	var out []string
	for len(r) > width {
		cut := width
		for i := cut; i > width/2; i-- {
			if r[i] == ' ' {
				cut = i
				break
			}
		}
		out = append(out, string(r[:cut]))
		r = r[cut:]
		for len(r) > 0 && r[0] == ' ' {
			r = r[1:]
		}
	}
	out = append(out, string(r))
	return out
}

func clip(s string, width int) string {
	if width < 4 || lipgloss.Width(s) <= width {
		return s
	}
	r := []rune(s)
	for len(r) > 0 && lipgloss.Width(string(r)) > width-1 {
		r = r[:len(r)-1]
	}
	return string(r) + "…"
}

func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.TrimSpace(s)
}
