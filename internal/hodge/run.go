package hodge

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"modelrun/internal/catalog"
	"modelrun/internal/chatfmt"
	"modelrun/internal/mark"
	"modelrun/internal/tokenizers"
)

// Turn is one chat message.
type Turn struct {
	Role    string
	Content string
}

// Opt is the sampler. ConfigSample fills any of Temp and TopP the caller left unset
// from the checkpoint's generation_config. Temp 0 is greedy. TempSet and TopSet
// mark a value the caller chose, including a temperature of 0.
type Opt struct {
	N            int
	Temp         float64
	TopP         float64
	ConfigSample bool
	TempSet      bool
	TopSet       bool
}

type Model struct {
	name       string
	sys        string
	D, F, L    int
	H, KV, HD  int
	Vocab      int
	eps, theta float64
	act        string
	gemma      bool
	embedScale bool
	queryScale float64
	attnCap    float64
	finalCap   float64
	ropeScale  int
	ropeFactor float64
	ropeLow    float64
	ropeHigh   float64
	ropeOrig   float64
	temp, topP float64
	embed      []byte
	final      []float32
	layers     []layer
	freq       []float64
	gpu        *gpuRT
	tok        *tokenizers.Tokenizer
	tokPath    string
	tokCfg     string
	bos        []int
	stop       map[int]bool
	kind       chatfmt.Kind
}

type dialog struct {
	kc, vc [][][]float32
	pos    int
	lg     []float32
}

func systemText(name string) string {
	return "You are " + name + ", a helpful assistant that runs on this device. Answer the person's question. State a fact only when you are sure of it. When you are not sure, say so in one sentence. Keep the answer short unless the person asks for a long one. When they want a recipe, give the ingredients and the method. When the person asks you to run a command or look at files, reply with only {\"type\":\"tool_call\",\"tool\":\"exec\",\"args\":{\"cmd\":\"ls\"}}. That command runs in the TunnelTug guest, never on this machine. After the results, answer in plain words."
}

// ErrInterrupted is a turn stopped by Ctrl+C.
var ErrInterrupted = errors.New("interrupted")

var interrupted atomic.Bool

// Interrupt stops the token loop and the sandbox call.
func Interrupt() { interrupted.Store(true) }

// ClearInterrupt arms a new turn.
func ClearInterrupt() { interrupted.Store(false) }

// Interrupted reports a Ctrl+C since the last clear.
func Interrupted() bool { return interrupted.Load() }

// ropeOf reads rope_scaling the way rope.wgsl names it: 0 none, 1 linear, 2 llama3.
func ropeOf(raw json.RawMessage) (scale int, factor, low, high, orig float64) {
	factor, low, high, orig = 1, 1, 4, 8192
	if len(raw) == 0 || string(raw) == "null" {
		return 0, factor, low, high, orig
	}
	var o struct {
		Type   string  `json:"rope_type"`
		Factor float64 `json:"factor"`
		Low    float64 `json:"low_freq_factor"`
		High   float64 `json:"high_freq_factor"`
		Orig   float64 `json:"original_max_position_embeddings"`
	}
	if json.Unmarshal(raw, &o) != nil {
		return 0, factor, low, high, orig
	}
	switch o.Type {
	case "linear":
		scale = 1
	case "llama3":
		scale = 2
	}
	if o.Factor > 0 {
		factor = o.Factor
	}
	if o.Low > 0 {
		low = o.Low
	}
	if o.High > 0 {
		high = o.High
	}
	if o.Orig > 0 {
		orig = o.Orig
	}
	return scale, factor, low, high, orig
}

func open(meta catalog.Model) (*Model, error) {
	if meta.Config == "" || meta.Tokenizer == "" || meta.TokenizerConfig == "" {
		return nil, fmt.Errorf("%s: config, tokenizer, and tokenizer_config are set in the config file", meta.Name)
	}
	cfgb, err := os.ReadFile(meta.Config)
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Hidden     int             `json:"hidden_size"`
		Inter      int             `json:"intermediate_size"`
		Layers     int             `json:"num_hidden_layers"`
		Heads      int             `json:"num_attention_heads"`
		KV         int             `json:"num_key_value_heads"`
		HD         int             `json:"head_dim"`
		Vocab      int             `json:"vocab_size"`
		Eps        float64         `json:"rms_norm_eps"`
		Theta      float64         `json:"rope_theta"`
		Act        string          `json:"hidden_act"`
		Act2       string          `json:"hidden_activation"`
		Tie        *bool           `json:"tie_word_embeddings"`
		RMS        string          `json:"rms_norm"`
		QueryScale float64         `json:"query_pre_attn_scalar"`
		AttnCap    float64         `json:"attn_logit_softcapping"`
		FinalCap   float64         `json:"final_logit_softcapping"`
		BOS        *int            `json:"bos_token_id"`
		Rope       json.RawMessage `json:"rope_scaling"`
	}
	if err := json.Unmarshal(cfgb, &cfg); err != nil {
		return nil, err
	}
	if cfg.Tie != nil && !*cfg.Tie {
		return nil, fmt.Errorf("%s: Hodge ties the head to the embedding", meta.Name)
	}
	if cfg.KV == 0 {
		cfg.KV = cfg.Heads
	}
	if cfg.HD == 0 && cfg.Heads > 0 {
		cfg.HD = cfg.Hidden / cfg.Heads
	}
	if cfg.Hidden < 4 || cfg.Inter < 4 || cfg.Layers < 1 || cfg.Heads < 1 || cfg.Vocab < 1 || cfg.HD < 2 {
		return nil, fmt.Errorf("%s: config does not describe a hodge checkpoint", meta.Name)
	}
	if cfg.Heads%cfg.KV != 0 || cfg.Hidden%4 != 0 || cfg.Inter%4 != 0 || cfg.HD%2 != 0 {
		return nil, fmt.Errorf("%s: head or width is not a multiple the kernels accept", meta.Name)
	}
	act := cfg.Act
	if cfg.Act2 != "" {
		act = cfg.Act2
	}
	if act == "" {
		act = "relu2"
	}
	if gateFn(act) == nil {
		return nil, fmt.Errorf("%s: hidden activation %q is not in the runtime", meta.Name, act)
	}
	eps := cfg.Eps
	if eps == 0 {
		eps = 1e-5
	}
	theta := cfg.Theta
	if theta == 0 {
		theta = 500000
	}
	name := meta.Detail
	if name == "" {
		name = meta.Name
	}
	m := &Model{
		name: name, sys: systemText(name),
		D: cfg.Hidden, F: cfg.Inter, L: cfg.Layers,
		H: cfg.Heads, KV: cfg.KV, HD: cfg.HD, Vocab: cfg.Vocab,
		eps: eps, theta: theta, act: act,
		gemma:      cfg.Act2 != "" || cfg.RMS == "gemma",
		queryScale: cfg.QueryScale, attnCap: cfg.AttnCap, finalCap: cfg.FinalCap,
		temp: 0.6, topP: 0.9, stop: map[int]bool{},
		tokPath: meta.Tokenizer, tokCfg: meta.TokenizerConfig,
	}
	m.ropeScale, m.ropeFactor, m.ropeLow, m.ropeHigh, m.ropeOrig = ropeOf(cfg.Rope)
	if cfg.BOS != nil && *cfg.BOS >= 0 {
		m.bos = []int{*cfg.BOS}
	}
	m.embedScale = m.gemma
	if meta.GenerationConfig != "" {
		b, err := os.ReadFile(meta.GenerationConfig)
		if err != nil {
			return nil, err
		}
		var g struct {
			EOS  json.RawMessage `json:"eos_token_id"`
			Temp float64         `json:"temperature"`
			TopP float64         `json:"top_p"`
		}
		if json.Unmarshal(b, &g) == nil {
			if g.Temp > 0 {
				m.temp = g.Temp
			}
			if g.TopP > 0 {
				m.topP = g.TopP
			}
			m.addEOS(g.EOS)
		}
	}
	logf("loading %s (%s)", meta.Name, name)
	raw, err := decodeAll(meta.Path)
	if err != nil {
		return nil, err
	}
	if layerCount(raw) != m.L {
		return nil, fmt.Errorf("%s: config says %d layers, the checkpoint has %d", meta.Name, m.L, layerCount(raw))
	}
	m.embed, err = assembleEmbed(raw, m.Vocab, m.D)
	if err != nil {
		return nil, err
	}
	if m.final, err = vec(raw, "model.norm.weight", m.D, false); err != nil {
		return nil, err
	}
	A := m.H * m.HD
	m.layers = make([]layer, m.L)
	for i := 0; i < m.L; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		L := &m.layers[i]
		if L.attnN, err = vec(raw, p+"input_layernorm.weight", m.D, false); err != nil {
			return nil, err
		}
		if L.ffnN, err = vec(raw, ffnNormName(raw, p), m.D, false); err != nil {
			return nil, err
		}
		if L.attnS, err = vec(raw, p+"self_attn.attn_sub_norm.weight", A, true); err != nil {
			return nil, err
		}
		if L.ffnS, err = vec(raw, p+"mlp.ffn_sub_norm.weight", m.F, true); err != nil {
			return nil, err
		}
		if L.qn, err = vec(raw, p+"self_attn.q_norm.weight", m.HD, true); err != nil {
			return nil, err
		}
		if L.kn, err = vec(raw, p+"self_attn.k_norm.weight", m.HD, true); err != nil {
			return nil, err
		}
		if L.q, err = buildLin(raw, p+"self_attn.q_proj", A, m.D); err != nil {
			return nil, err
		}
		if L.k, err = buildLin(raw, p+"self_attn.k_proj", m.KV*m.HD, m.D); err != nil {
			return nil, err
		}
		if L.v, err = buildLin(raw, p+"self_attn.v_proj", m.KV*m.HD, m.D); err != nil {
			return nil, err
		}
		if L.o, err = buildLin(raw, p+"self_attn.o_proj", m.D, A); err != nil {
			return nil, err
		}
		if L.gate, err = buildLin(raw, p+"mlp.gate_proj", m.F, m.D); err != nil {
			return nil, err
		}
		if L.up, err = buildLin(raw, p+"mlp.up_proj", m.F, m.D); err != nil {
			return nil, err
		}
		if L.down, err = buildLin(raw, p+"mlp.down_proj", m.D, m.F); err != nil {
			return nil, err
		}
		logf("layer %d  q %s  down %s", i, linKind(L.q), linKind(L.down))
	}
	m.freq = make([]float64, m.HD/2)
	for d := range m.freq {
		m.freq[d] = math.Pow(m.theta, -2*float64(d)/float64(m.HD))
	}
	tok, err := tokenizers.Load(meta.Tokenizer)
	if err != nil {
		return nil, err
	}
	m.tok = tok
	m.learnFormat(meta.TokenizerConfig)
	if err := m.boot(); err != nil {
		return nil, err
	}
	logf("ready %s  %d layers  hidden %d", meta.Name, m.L, m.D)
	return m, nil
}

func (m *Model) addEOS(raw json.RawMessage) {
	if len(raw) == 0 {
		return
	}
	var n float64
	if json.Unmarshal(raw, &n) == nil {
		m.stop[int(n)] = true
		return
	}
	var arr []float64
	if json.Unmarshal(raw, &arr) == nil {
		for _, v := range arr {
			m.stop[int(v)] = true
		}
	}
}

func chatTemplates(path string) (string, []chatfmt.Listed) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", nil
	}
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return "", nil
	}
	root, _ := v.(map[string]any)
	return chatfmt.ReadTemplates(root["chat_template"])
}

// learnFormat is the site's chat choice: an exact 2B4T template stays the role
// format even when the file also carries Llama 3 header pieces.
func (m *Model) learnFormat(tokenizerConfig string) {
	primary, listed := chatTemplates(tokenizerConfig)
	has := func(piece string) bool { return m.tok.ID(piece) >= 0 }
	have := chatfmt.Available(has, primary, listed, "")
	m.kind = chatfmt.Plain
	if len(have) > 0 {
		m.kind = have[0]
	}
	for _, piece := range chatfmt.Stops(m.kind) {
		if id := m.tok.ID(piece); id >= 0 {
			m.stop[id] = true
		}
	}
	logf("chat %s", chatfmt.Name(m.kind))
}

func (m *Model) pieces(s string) []int {
	if s == "" {
		return nil
	}
	return m.tok.Encode(s, true)
}

func (m *Model) show(ids []int) string {
	return m.tok.Decode(ids, true)
}

// turnIDs is one user turn the way hkruntime renders it. first carries BOS and the system text.
func turnIDs(tok *tokenizers.Tokenizer, kind chatfmt.Kind, bos []int, system, user string, first bool) []int {
	var ids []int
	if first {
		ids = append(ids, bos...)
	}
	return append(ids, tok.Encode(chatfmt.Render(kind, system, user, first), true)...)
}

func (m *Model) stopPiece() string {
	for _, piece := range chatfmt.Stops(m.kind) {
		if m.tok.ID(piece) >= 0 {
			return piece
		}
	}
	return ""
}

func quant(y []float32) ([]int8, float32) {
	mx := float32(1e-5)
	for _, v := range y {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	s := 127 / mx
	q := make([]int8, len(y))
	for i, v := range y {
		q[i] = int8(math.Max(-128, math.Min(127, math.Round(float64(v*s)))))
	}
	return q, 1 / s
}

func rms(x, w []float32, eps float64, gemma bool) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+eps))
	o := make([]float32, len(x))
	for i := range x {
		g := w[i]
		if gemma {
			g = 1 + g
		}
		o[i] = x[i] * inv * g
	}
	return o
}

func headNorm(x, w []float32, hd int, eps float64, gemma bool) {
	for h := 0; h+hd <= len(x); h += hd {
		copy(x[h:h+hd], rms(x[h:h+hd], w, eps, gemma))
	}
}

// ffnNormName is the FFN pre-norm. Llama and Qwen store it as post_attention_layernorm.
// VaultGemma stores that same pre-norm as pre_feedforward_layernorm.
func ffnNormName(t map[string]got, prefix string) string {
	post := prefix + "post_attention_layernorm.weight"
	if _, ok := t[post]; ok {
		return post
	}
	pre := prefix + "pre_feedforward_layernorm.weight"
	if _, ok := t[pre]; ok {
		return pre
	}
	return post
}

func linKind(l lin) string {
	if l.f != nil {
		if len(l.bias) > 0 {
			return "float+bias"
		}
		return "float"
	}
	return fmt.Sprintf("limbs %d", len(l.limbs))
}

// actID is norm.wgsl's ACT override. 0 relu^2, 1 silu, 2 gelu tanh.
func actID(act string) (int, error) {
	switch act {
	case "relu2":
		return 0, nil
	case "silu", "swish":
		return 1, nil
	case "gelu_pytorch_tanh", "gelu_new":
		return 2, nil
	default:
		return 0, fmt.Errorf("hidden activation %q is not in norm.wgsl", act)
	}
}

func gateFn(act string) func(float32) float32 {
	switch act {
	case "relu2":
		return func(g float32) float32 {
			if g < 0 {
				g = 0
			}
			return g * g
		}
	case "silu", "swish":
		return func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }
	case "gelu_pytorch_tanh", "gelu_new":
		return func(g float32) float32 {
			x := float64(g)
			return float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
		}
	default:
		return nil
	}
}

func (m *Model) row(id int) []float32 {
	off := id * m.D * 2
	o := make([]float32, m.D)
	for j := 0; j < m.D; j++ {
		o[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(m.embed[off+2*j:])) << 16)
	}
	if m.embedScale {
		sc := float32(math.Sqrt(float64(m.D)))
		for j := range o {
			o[j] *= sc
		}
	}
	return o
}

func (m *Model) rope(x []float32, pos int) {
	hd := m.HD
	half := hd / 2
	for h := 0; h+hd <= len(x); h += hd {
		for d := 0; d < half; d++ {
			ang := float64(pos) * m.freq[d]
			c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
			a, b := x[h+d], x[h+d+half]
			x[h+d] = a*c - b*s
			x[h+d+half] = a*s + b*c
		}
	}
}

func (m *Model) linear(l lin, q []int8, xs float32) ([]float32, error) {
	return nil, fmt.Errorf("linear %d is on the device", l.n)
}

func (m *Model) attend(q []float32, kc, vc [][]float32) []float32 {
	H, KV, HD := m.H, m.KV, m.HD
	ao := make([]float32, H*HD)
	T := len(kc)
	den := math.Sqrt(float64(HD))
	if m.queryScale > 0 {
		den = math.Sqrt(m.queryScale)
	}
	for h := 0; h < H; h++ {
		g := h / (H / KV)
		sc := make([]float64, T)
		mx := -1e30
		for t := 0; t < T; t++ {
			var a float64
			for d := 0; d < HD; d++ {
				a += float64(q[h*HD+d] * kc[t][g*HD+d])
			}
			a /= den
			if m.attnCap > 0 {
				a = math.Tanh(a/m.attnCap) * m.attnCap
			}
			sc[t] = a
			if a > mx {
				mx = a
			}
		}
		var z float64
		for t := range sc {
			sc[t] = math.Exp(sc[t] - mx)
			z += sc[t]
		}
		for t := range sc {
			w := float32(sc[t] / z)
			for d := 0; d < HD; d++ {
				ao[h*HD+d] += w * vc[t][g*HD+d]
			}
		}
	}
	return ao
}

func (m *Model) logits(x []float32) []float32 {
	y := rms(x, m.final, m.eps, m.gemma)
	out := make([]float32, m.Vocab)
	emb := m.embed
	D := m.D
	workers := 8
	var wg sync.WaitGroup
	for p := 0; p < workers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for r := p; r < len(out); r += workers {
				var a float32
				off := r * D * 2
				for j := 0; j < D; j++ {
					a += y[j] * math.Float32frombits(uint32(binary.LittleEndian.Uint16(emb[off+2*j:]))<<16)
				}
				out[r] = a
			}
		}(p)
	}
	wg.Wait()
	if m.finalCap > 0 {
		cap := m.finalCap
		for i, v := range out {
			out[i] = float32(math.Tanh(float64(v)/cap) * cap)
		}
	}
	return out
}

func (m *Model) feed(d *dialog, id int, head bool) error {
	if id < 0 || id >= m.Vocab {
		return fmt.Errorf("token %d is outside the embedding", id)
	}
	return m.step(d, []int{id}, head)
}

func (m *Model) prefill(d *dialog, ids []int) error {
	if len(ids) == 0 {
		return fmt.Errorf("empty prompt")
	}
	tail := ids
	if len(tail) > 24 {
		tail = tail[len(tail)-24:]
	}
	logf("prompt %d  tail %q", len(ids), m.show(tail))
	logf("prefill %d tokens", len(ids))
	t0 := time.Now()
	if err := m.step(d, ids, true); err != nil {
		return err
	}
	logf("prefill %d/%d  %s", len(ids), len(ids), time.Since(t0).Round(time.Millisecond))
	return nil
}

func emitText(pending *[]byte, piece string, emit func(string)) {
	if emit == nil {
		return
	}
	*pending = append(*pending, piece...)
	i := 0
	for i < len(*pending) {
		r, size := utf8.DecodeRune((*pending)[i:])
		if r == utf8.RuneError && size == 1 {
			break
		}
		i += size
	}
	if i > 0 {
		emit(string((*pending)[:i]))
		*pending = (*pending)[i:]
	}
}

func (m *Model) generate(d *dialog, opt Opt, emit func(string)) (string, error) {
	// N <= 0 runs until an end token or the context window. A positive N is a cap.
	n := opt.N
	if n <= 0 {
		n = 1 << 30
	}
	temp, top := opt.Temp, opt.TopP
	if opt.ConfigSample {
		if !opt.TempSet {
			temp = m.temp
		}
		if !opt.TopSet {
			top = m.topP
		}
	}
	var rng *rand.Rand
	if temp > 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	var pending []byte
	var all []byte
	var recent []int
	ended := false
	t0 := time.Now()
	keepID := func(id int) bool {
		return m.stop[id] || (m.tok != nil && m.tok.IsSpecial(id))
	}
	for j := 0; j < n; j++ {
		if interrupted.Load() {
			return string(all), ErrInterrupted
		}
		if m.gpu != nil && d.pos >= m.gpu.ctx {
			break
		}
		id := draw(d.lg, recent, temp, top, keepID, func(id int) bool { return m.stop[id] }, rng)
		if m.stop[id] {
			ended = true
			if err := m.feed(d, id, true); err != nil {
				return "", err
			}
			break
		}
		piece := string(m.tok.Bytes(id))
		all = append(all, piece...)
		recent = remember(recent, id, repeatLast)
		emitText(&pending, piece, emit)
		if err := m.feed(d, id, true); err != nil {
			return "", err
		}
	}
	if len(pending) > 0 && emit != nil {
		emit(string(pending))
	}
	if !ended {
		if end := m.stopPiece(); end != "" {
			ids := m.pieces(end)
			for i, id := range ids {
				if err := m.feed(d, id, i == len(ids)-1); err != nil {
					return "", err
				}
			}
		}
	}
	secs := time.Since(t0).Seconds()
	if secs > 0 {
		logf("generated in %s", time.Since(t0).Round(time.Millisecond))
	}
	return string(all), nil
}

func (m *Model) refuse(texts ...string) error {
	c, err := mark.Cached(m.tokPath, m.tokCfg)
	if err != nil || c == nil {
		return err
	}
	for _, text := range texts {
		if err := c.Refuse(text); err != nil {
			return err
		}
	}
	return nil
}

func (m *Model) transcript(turns []Turn, opt Opt, emit func(string)) (string, error) {
	var texts []string
	for _, t := range turns {
		if strings.EqualFold(t.Role, "system") {
			continue
		}
		texts = append(texts, t.Content)
	}
	if err := m.refuse(texts...); err != nil {
		return "", err
	}
	d := &dialog{}
	sys := systemFor(m)
	for _, t := range turns {
		if strings.EqualFold(t.Role, "system") && strings.TrimSpace(t.Content) != "" {
			sys = t.Content
		}
	}
	var ids []int
	first := true
	for _, t := range turns {
		switch strings.ToLower(t.Role) {
		case "system":
		case "user":
			use := ""
			if first {
				use = sys
			}
			ids = append(ids, turnIDs(m.tok, m.kind, m.bos, use, t.Content, first)...)
			first = false
		case "assistant":
			ids = append(ids, m.pieces(t.Content)...)
			if end := m.stopPiece(); end != "" {
				ids = append(ids, m.pieces(end)...)
			}
		default:
			return "", fmt.Errorf("role %q", t.Role)
		}
	}
	if first {
		return "", fmt.Errorf("empty prompt")
	}
	if err := m.prefill(d, ids); err != nil {
		return "", err
	}
	return m.generate(d, opt, emit)
}

func (m *Model) raw(prompt string, opt Opt, emit func(string)) (string, error) {
	if err := m.refuse(prompt); err != nil {
		return "", err
	}
	d := &dialog{}
	ids := append(append([]int{}, m.bos...), m.tok.Encode(prompt, true)...)
	if err := m.prefill(d, ids); err != nil {
		return "", err
	}
	return m.generate(d, opt, emit)
}

var (
	liveMu   sync.Mutex
	live     *Model
	liveName string
	chat     *dialog
	chatName string
)

func get(name string) (*Model, error) {
	if live != nil {
		if liveName == name {
			return live, nil
		}
		return nil, fmt.Errorf("%s is loaded; quit and run the other model", liveName)
	}
	meta, err := catalog.Load(name)
	if err != nil {
		return nil, err
	}
	if meta.Family != catalog.FamilyHodge {
		return nil, fmt.Errorf("no hodge model %q", name)
	}
	m, err := open(meta)
	if err != nil {
		return nil, err
	}
	live, liveName = m, name
	chat, chatName = nil, ""
	return m, nil
}

// Prepare loads the checkpoint and leaves the weights on the device.
func Prepare(name string) error {
	liveMu.Lock()
	defer liveMu.Unlock()
	_, err := get(name)
	return err
}

var skillBody string

// UseSkill arms one skill for the next chat. An empty body turns it off.
// The chat cache is dropped so the next turn sees the skill. The weights stay.
func UseSkill(body string) {
	liveMu.Lock()
	skillBody = strings.TrimSpace(body)
	chat, chatName = nil, ""
	liveMu.Unlock()
}

func systemFor(m *Model) string {
	if skillBody == "" {
		return m.sys
	}
	return m.sys + "\n\n" + skillBody
}

// Reset drops the chat cache. The checkpoint stays loaded.
func Reset(name string) {
	liveMu.Lock()
	defer liveMu.Unlock()
	if name == "" || name == chatName {
		chat, chatName = nil, ""
	}
}

// Transcript is one stateless chat. turns keeps the client's history.
func Transcript(name string, turns []Turn, opt Opt, emit func(string)) (string, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	m, err := get(name)
	if err != nil {
		return "", err
	}
	return m.transcript(turns, opt, emit)
}

// Raw is a completion of prompt, without the chat template.
func Raw(name, prompt string, opt Opt, emit func(string)) (string, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	m, err := get(name)
	if err != nil {
		return "", err
	}
	return m.raw(prompt, opt, emit)
}

// Absorb writes one finished exchange into the chat cache. A looked-up answer
// is the page's own sentences, so it is stored here and the model is not asked
// to rewrite it.
func Absorb(name, user, assistant string) error {
	liveMu.Lock()
	defer liveMu.Unlock()
	m, err := get(name)
	if err != nil {
		return err
	}
	if err := m.refuse(user, assistant); err != nil {
		return err
	}
	if chat == nil || chatName != name {
		chat = &dialog{}
		chatName = name
	}
	var ids []int
	if chat.pos == 0 {
		ids = turnIDs(m.tok, m.kind, m.bos, systemFor(m), user, true)
	} else {
		ids = turnIDs(m.tok, m.kind, nil, "", user, false)
	}
	ids = append(ids, m.pieces(assistant)...)
	if end := m.stopPiece(); end != "" {
		ids = append(ids, m.pieces(end)...)
	}
	if len(ids) == 0 {
		return nil
	}
	return m.prefill(chat, ids)
}

// Continue is the TUI conversation. The first call for a model starts a new chat.
func Continue(name, user string, opt Opt, emit func(string)) (string, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	m, err := get(name)
	if err != nil {
		return "", err
	}
	if chat == nil || chatName != name {
		chat = &dialog{}
		chatName = name
	}
	if err := m.refuse(user); err != nil {
		return "", err
	}
	var ids []int
	if chat.pos == 0 {
		ids = turnIDs(m.tok, m.kind, m.bos, systemFor(m), user, true)
	} else {
		ids = turnIDs(m.tok, m.kind, nil, "", user, false)
	}
	if err := m.prefill(chat, ids); err != nil {
		return "", err
	}
	return m.generate(chat, opt, emit)
}

// Loaded is the checkpoint held in this process, if any.
func Loaded() (string, bool) {
	liveMu.Lock()
	defer liveMu.Unlock()
	if live == nil {
		return "", false
	}
	return liveName, true
}
