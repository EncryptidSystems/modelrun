package hodge

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"modelrun/internal/paths"
	"modelrun/internal/rt"
)

// gpuRT is the resident device. Every weight is uploaded once. A token is
// one recorded command buffer, not a fresh upload of each matrix.
type gpuRT struct {
	dev            *rt.Dev
	ctx            int
	stepID, steps  uint32
	logits         uint32
	embedP         uint32
	embedX         uint32
	chunks         []embChunk
	ropeP, attnP   uint32
	ropeX, attnX   uint32
	wedgeP, wedgeG uint32
	normD, normF   uint32
	normFin        uint32
	subP           uint32 // attention sub-norm; normD when H*HD equals hidden
	fin            uint32
	layers         []gpuLayer
	hl             map[string]uint32
}

type gpuLayer struct {
	attnN, ffnN, sub, ffnS uint32
	rope, attn             uint32
	q, k, v, o             linG
	gate, up, down         linG
}

// embChunk is one slice of the tied embedding. embed.wgsl reads st.row inside
// this chunk, and lmhead.wgsl writes this chunk's rows of logits.
type embChunk struct {
	start  int
	rows   int
	embedG uint32
	lmP    uint32
	lmG    uint32
	lmX    uint32
}

type linG struct {
	pipe []uint32
	grp  []uint32
	x    uint32
}

func (m *Model) boot() error {
	if m.qnAny() {
		return fmt.Errorf("%s: q/k norm is not in this Dawn step", m.name)
	}
	dev, err := rt.Open()
	if err != nil {
		return err
	}
	g := &gpuRT{dev: dev, ctx: 4096, hl: map[string]uint32{}}
	k := paths.Default().KernelsDir()
	read := func(name string) (string, error) {
		b, err := os.ReadFile(filepath.Join(k, name))
		return string(b), err
	}
	dot, err := read("dot4_soft.wgsl")
	if err != nil {
		dev.Close()
		return err
	}
	lin, err := read("hodgelinear.wgsl")
	if err != nil {
		dev.Close()
		return err
	}
	load := func(name, src string) error { return dev.Shader(name, src) }
	if err = load("hl", dot+"\n"+lin); err != nil {
		dev.Close()
		return err
	}
	names := []string{"embed.wgsl", "norm.wgsl", "rope.wgsl", "attention.wgsl", "lmhead.wgsl", "wedge.wgsl"}
	if m.floatAny() {
		names = append(names, "dense.wgsl")
	}
	for _, n := range names {
		s, err := read(n)
		if err != nil {
			dev.Close()
			return err
		}
		if err = load(n, s); err != nil {
			dev.Close()
			return err
		}
	}
	if err = g.alloc(m); err != nil {
		dev.Close()
		return err
	}
	m.gpu = g
	m.embed = nil
	logf("weights resident  %d layers  ctx %d", m.L, g.ctx)
	return nil
}

func (m *Model) floatAny() bool {
	for i := range m.layers {
		L := &m.layers[i]
		if L.q.f != nil || L.k.f != nil || L.v.f != nil || L.o.f != nil || L.gate.f != nil || L.up.f != nil || L.down.f != nil {
			return true
		}
	}
	return false
}

func (m *Model) qnAny() bool {
	for i := range m.layers {
		if m.layers[i].qn != nil || m.layers[i].kn != nil {
			return true
		}
	}
	return false
}

func (g *gpuRT) alloc(m *Model) error {
	d := g.dev
	maxK := m.D
	if m.F > maxK {
		maxK = m.F
	}
	A := m.H * m.HD
	if A > maxK {
		maxK = A
	}
	zero := func(name string, n int) error {
		_, err := d.Buf(name, nil, n)
		return err
	}
	if err := zero("x", m.D*4); err != nil {
		return err
	}
	if err := zero("ao", A*4); err != nil {
		return err
	}
	if err := zero("qf", A*4); err != nil {
		return err
	}
	if err := zero("kf", m.KV*m.HD*4); err != nil {
		return err
	}
	if err := zero("vf", m.KV*m.HD*4); err != nil {
		return err
	}
	if err := zero("gv", m.F*4); err != nil {
		return err
	}
	if err := zero("uv", m.F*4); err != nil {
		return err
	}
	if err := zero("wr", m.D*4); err != nil {
		return err
	}
	if err := zero("y", maxK*4); err != nil {
		return err
	}
	if err := zero("xq", maxK); err != nil {
		return err
	}
	if err := zero("xs", 16); err != nil {
		return err
	}
	if err := zero("att", m.H*g.ctx*4); err != nil {
		return err
	}
	if err := zero("flag", 16); err != nil {
		return err
	}
	per := g.ctx * m.KV * m.HD * 4
	if err := zero("kc", m.L*per); err != nil {
		return err
	}
	if err := zero("vc", m.L*per); err != nil {
		return err
	}
	var err error
	if g.stepID, err = d.Buf("step", make([]byte, 16), 256); err != nil {
		return err
	}
	if g.steps, err = d.Buf("steps", make([]byte, 16), 128*16); err != nil {
		return err
	}
	if g.logits, err = d.Buf("logits", nil, m.Vocab*4); err != nil {
		return err
	}
	ones := make([]float32, maxK)
	for i := range ones {
		ones[i] = 1
	}
	if _, err = d.Buf("ones", f32b(ones), len(ones)*4); err != nil {
		return err
	}
	if m.floatAny() {
		if _, err = d.Buf("zbias", []byte{0, 0, 0, 0}, 4); err != nil {
			return err
		}
	}
	ord := make([]byte, m.D*4)
	for i := 0; i < m.D; i++ {
		binary.LittleEndian.PutUint32(ord[i*4:], uint32(i))
	}
	if _, err = d.Buf("order", ord, len(ord)); err != nil {
		return err
	}
	logf("uploading embedding  %d bytes", len(m.embed))
	if err = g.uploadEmbed(m); err != nil {
		return err
	}
	if _, err = d.Buf("final", f32b(m.final), len(m.final)*4); err != nil {
		return err
	}

	scale := "1"
	if m.embedScale {
		scale = strconv.FormatFloat(math.Sqrt(float64(m.D)), 'f', -1, 64)
	}
	uni := []byte{rt.KindUni, rt.KindRO, rt.KindRW}
	if g.embedP, err = d.Pipe("embed", "embed.wgsl", "main", "D="+strconv.Itoa(m.D)+" SCALE="+scale, uni); err != nil {
		return err
	}
	for i := range g.chunks {
		c := &g.chunks[i]
		name := fmt.Sprintf("emb%d", c.start)
		if c.embedG, err = d.Group(name, "embed", []rt.Bind{{"step", 0, 16}, {name, 0, 0}, {"x", 0, 0}}); err != nil {
			return err
		}
	}
	g.embedX = uint32((m.D + 255) / 256)

	act, err := actID(m.act)
	if err != nil {
		return err
	}
	nk := []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRW}
	eps := strconv.FormatFloat(m.eps, 'g', -1, 64)
	gem := "0"
	if m.gemma {
		gem = "1"
	}
	base := "K=%d EPS=" + eps + " GEMMA=" + gem
	if g.normD, err = d.Pipe("normD", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=0 ACT=0", m.D), nk); err != nil {
		return err
	}
	if g.normF, err = d.Pipe("normF", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=1 ACT=%d", m.F, act), nk); err != nil {
		return err
	}
	if g.normFin, err = d.Pipe("normFin", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=2 ACT=0", m.D), nk); err != nil {
		return err
	}
	// The attention sub-norm covers ao, which is H*HD wide. normD is compiled
	// at the hidden size. Sharing it when those differ reads past ao.
	g.subP = g.normD
	subPipe := "normD"
	if A != m.D {
		if A%4 != 0 {
			return fmt.Errorf("attention width %d is not a multiple of 4", A)
		}
		if g.subP, err = d.Pipe("normA", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=0 ACT=0", A), nk); err != nil {
			return err
		}
		subPipe = "normA"
	}
	ropeOv := fmt.Sprintf("H=%d KVH=%d HD=%d THETA=%s SCALING=%d FACTOR=%s LOW_FF=%s HIGH_FF=%s ORIG=%s",
		m.H, m.KV, m.HD, strconv.FormatFloat(m.theta, 'g', -1, 64), m.ropeScale,
		strconv.FormatFloat(m.ropeFactor, 'g', -1, 64),
		strconv.FormatFloat(m.ropeLow, 'g', -1, 64),
		strconv.FormatFloat(m.ropeHigh, 'g', -1, 64),
		strconv.FormatFloat(m.ropeOrig, 'g', -1, 64))
	rk := []byte{rt.KindUni, rt.KindRW, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW}
	if g.ropeP, err = d.Pipe("rope", "rope.wgsl", "main", ropeOv, rk); err != nil {
		return err
	}
	g.ropeX = uint32((m.H*m.HD/2 + m.KV*m.HD/2 + 63) / 64)
	denom := "0"
	if m.queryScale > 0 {
		denom = strconv.FormatFloat(m.queryScale, 'g', -1, 64)
	}
	ak := []byte{rt.KindUni, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW}
	if g.attnP, err = d.Pipe("attn", "attention.wgsl", "main", fmt.Sprintf("H=%d KVH=%d HD=%d CTX=%d DENOM=%s", m.H, m.KV, m.HD, g.ctx, denom), ak); err != nil {
		return err
	}
	g.attnX = uint32(m.H)
	wk := []byte{rt.KindRW, rt.KindRO, rt.KindRO, rt.KindRW}
	if g.wedgeP, err = d.Pipe("wedge", "wedge.wgsl", "main", fmt.Sprintf("DIM=%d NCELL=%d M=1", m.D, m.D), wk); err != nil {
		return err
	}
	if g.wedgeG, err = d.Group("wedge", "wedge", []rt.Bind{{"x", 0, 0}, {"wr", 0, 0}, {"order", 0, 0}, {"flag", 0, 4}}); err != nil {
		return err
	}
	lk := []byte{rt.KindRO, rt.KindRO, rt.KindRW}
	for i := range g.chunks {
		c := &g.chunks[i]
		name := fmt.Sprintf("lm%d", c.start)
		if c.lmP, err = d.Pipe(name, "lmhead.wgsl", "main", fmt.Sprintf("D=%d ROWS=%d", m.D, c.rows), lk); err != nil {
			return err
		}
		if c.lmG, err = d.Group(name, name, []rt.Bind{
			{"y", 0, 0},
			{fmt.Sprintf("emb%d", c.start), 0, 0},
			{"logits", c.start * 4, c.rows * 4},
		}); err != nil {
			return err
		}
		c.lmX = uint32((c.rows + 7) / 8)
	}
	if g.fin, err = g.normGroup("final", "normFin", "x", "x", m.final, m.D); err != nil {
		return err
	}

	g.layers = make([]gpuLayer, m.L)
	for i := range m.layers {
		L := &m.layers[i]
		gl := &g.layers[i]
		if gl.attnN, err = g.normGroup(fmt.Sprintf("L%d-an", i), "normD", "x", "x", L.attnN, m.D); err != nil {
			return err
		}
		if gl.ffnN, err = g.normGroup(fmt.Sprintf("L%d-fn", i), "normD", "x", "x", L.ffnN, m.D); err != nil {
			return err
		}
		if gl.sub, err = g.normGroup(fmt.Sprintf("L%d-sub", i), subPipe, "ao", "ao", orOnes(L.attnS, A), orLen(L.attnS, A)); err != nil {
			return err
		}
		if gl.ffnS, err = g.normGroup(fmt.Sprintf("L%d-fs", i), "normF", "gv", "uv", orOnes(L.ffnS, m.F), orLen(L.ffnS, m.F)); err != nil {
			return err
		}
		if gl.q, err = g.linGroups(fmt.Sprintf("L%d-q", i), "qf", L.q); err != nil {
			return err
		}
		if gl.k, err = g.linGroups(fmt.Sprintf("L%d-k", i), "kf", L.k); err != nil {
			return err
		}
		if gl.v, err = g.linGroups(fmt.Sprintf("L%d-v", i), "vf", L.v); err != nil {
			return err
		}
		if gl.o, err = g.linGroups(fmt.Sprintf("L%d-o", i), "wr", L.o); err != nil {
			return err
		}
		if gl.gate, err = g.linGroups(fmt.Sprintf("L%d-g", i), "gv", L.gate); err != nil {
			return err
		}
		if gl.up, err = g.linGroups(fmt.Sprintf("L%d-u", i), "uv", L.up); err != nil {
			return err
		}
		if gl.down, err = g.linGroups(fmt.Sprintf("L%d-d", i), "wr", L.down); err != nil {
			return err
		}
		off := i * per
		if gl.rope, err = d.Group(fmt.Sprintf("L%d-rope", i), "rope", []rt.Bind{
			{"step", 0, 16}, {"qf", 0, 0}, {"kf", 0, 0}, {"vf", 0, 0},
			{"kc", off, per}, {"vc", off, per},
		}); err != nil {
			return err
		}
		if gl.attn, err = d.Group(fmt.Sprintf("L%d-attn", i), "attn", []rt.Bind{
			{"step", 0, 16}, {"qf", 0, 0}, {"kc", off, per}, {"vc", off, per}, {"att", 0, 0}, {"ao", 0, 0},
		}); err != nil {
			return err
		}
		L.q, L.k, L.v, L.o = lin{}, lin{}, lin{}, lin{}
		L.gate, L.up, L.down = lin{}, lin{}, lin{}
		L.attnN, L.ffnN, L.attnS, L.ffnS = nil, nil, nil, nil
		logf("layer %d resident", i)
	}
	m.final = nil
	return nil
}

func orOnes(w []float32, n int) []float32 {
	if w != nil {
		return w
	}
	return nil
}

func orLen(w []float32, n int) int {
	if w != nil {
		return len(w)
	}
	return n
}

func (g *gpuRT) normGroup(name, pipe, a, b string, w []float32, n int) (uint32, error) {
	buf := "ones"
	if w != nil {
		buf = name + "-w"
		if _, err := g.dev.Buf(buf, f32b(w), n*4); err != nil {
			return 0, err
		}
	}
	return g.dev.Group(name, pipe, []rt.Bind{
		{a, 0, 0}, {b, 0, 0}, {buf, 0, n * 4}, {"xq", 0, 0}, {"xs", 0, 16}, {"y", 0, 0},
	})
}

func (g *gpuRT) linGroups(name, dest string, l lin) (linG, error) {
	if l.f != nil {
		return g.denseGroups(name, dest, l)
	}
	var out linG
	out.x = uint32((l.n/4 + 7) / 8)
	parts := []struct {
		w     []byte
		s     float32
		c     int
		accum bool
	}{{l.w, l.s, 1, false}}
	for _, lb := range l.limbs {
		parts = append(parts, struct {
			w     []byte
			s     float32
			c     int
			accum bool
		}{lb.w, lb.s, lb.center, true})
	}
	for i, p := range parts {
		wn := fmt.Sprintf("%s-w%d", name, i)
		sn := fmt.Sprintf("%s-s%d", name, i)
		if _, err := g.dev.Buf(wn, p.w, len(p.w)); err != nil {
			return linG{}, err
		}
		sb := f32b([]float32{p.s})
		if _, err := g.dev.Buf(sn, sb, 16); err != nil {
			return linG{}, err
		}
		acc := "false"
		if p.accum {
			acc = "true"
		}
		key := fmt.Sprintf("K=%d NP=%d ACCUM=%s CENTER=%d", l.k, l.n/4, acc, p.c)
		id, ok := g.hl[key]
		if !ok {
			var err error
			id, err = g.dev.Pipe("hl-"+key, "hl", "main", key, []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW})
			if err != nil {
				return linG{}, err
			}
			g.hl[key] = id
		}
		gn := fmt.Sprintf("%s-g%d", name, i)
		gid, err := g.dev.Group(gn, "hl-"+key, []rt.Bind{
			{"xq", 0, 0}, {"xs", 0, 16}, {wn, 0, 0}, {sn, 0, 16}, {dest, 0, 0},
		})
		if err != nil {
			return linG{}, err
		}
		out.pipe = append(out.pipe, id)
		out.grp = append(out.grp, gid)
	}
	return out, nil
}

func (g *gpuRT) denseGroups(name, dest string, l lin) (linG, error) {
	if l.k%2 != 0 || len(l.f) != l.n*l.k*2 {
		return linG{}, fmt.Errorf("%s: dense weight %d bytes", name, len(l.f))
	}
	wn := name + "-w"
	if _, err := g.dev.Buf(wn, l.f, len(l.f)); err != nil {
		return linG{}, err
	}
	biasName := "zbias"
	bias := 0
	bb := l.bias
	if len(bb) > 0 {
		bias = 1
		biasName = name + "-b"
		if len(bb)%4 != 0 {
			bb = append(append([]byte{}, bb...), 0, 0)
		}
		if _, err := g.dev.Buf(biasName, bb, len(bb)); err != nil {
			return linG{}, err
		}
	}
	key := fmt.Sprintf("N=%d K=%d BIAS=%d", l.n, l.k, bias)
	id, ok := g.hl[key]
	if !ok {
		var err error
		id, err = g.dev.Pipe("dense-"+key, "dense.wgsl", "main", key, []byte{rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRO})
		if err != nil {
			return linG{}, err
		}
		g.hl[key] = id
	}
	gid, err := g.dev.Group(name, "dense-"+key, []rt.Bind{
		{"y", 0, 0}, {wn, 0, 0}, {dest, 0, 0}, {biasName, 0, 0},
	})
	if err != nil {
		return linG{}, err
	}
	return linG{pipe: []uint32{id}, grp: []uint32{gid}, x: uint32((l.n + 7) / 8)}, nil
}

func (m *Model) step(d *dialog, ids []int, head bool) error {
	if m.gpu == nil {
		return fmt.Errorf("no device")
	}
	const batch = 1
	for off := 0; off < len(ids); off += batch {
		if interrupted.Load() {
			return ErrInterrupted
		}
		n := len(ids) - off
		if n > batch {
			n = batch
		}
		last := head && off+n == len(ids)
		if err := m.stepN(d, ids[off:off+n], last); err != nil {
			return err
		}
	}
	return nil
}

func (m *Model) stepN(d *dialog, ids []int, head bool) error {
	g := m.gpu
	if d.pos+len(ids) > g.ctx {
		return fmt.Errorf("context full (%d)", g.ctx)
	}
	// The step uniform is written on the queue before each submit. A copy
	// inside the command buffer did not reach the embedding row or the position.
	for i, id := range ids {
		rec := make([]byte, 16)
		ch, row, err := g.chunk(id)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(rec[0:], uint32(d.pos))
		binary.LittleEndian.PutUint32(rec[4:], uint32(row))
		if err := g.dev.Write("step", 0, rec); err != nil {
			return err
		}
		ops := make([]rt.Op, 0, 8+m.L*16)
		ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: g.embedP, Group: ch.embedG, X: g.embedX, Y: 1, Z: 1})
		for li := range g.layers {
			L := &g.layers[li]
			ops = append(ops, disp(g.normD, L.attnN, 1)...)
			ops = append(ops, linOps(L.q)...)
			ops = append(ops, linOps(L.k)...)
			ops = append(ops, linOps(L.v)...)
			ops = append(ops, disp(g.ropeP, L.rope, g.ropeX)...)
			ops = append(ops, disp(g.attnP, L.attn, g.attnX)...)
			ops = append(ops, disp(g.subP, L.sub, 1)...)
			ops = append(ops, linOps(L.o)...)
			ops = append(ops, disp(g.wedgeP, g.wedgeG, 1)...)
			ops = append(ops, disp(g.normD, L.ffnN, 1)...)
			ops = append(ops, linOps(L.gate)...)
			ops = append(ops, linOps(L.up)...)
			ops = append(ops, disp(g.normF, L.ffnS, 1)...)
			ops = append(ops, linOps(L.down)...)
			ops = append(ops, disp(g.wedgeP, g.wedgeG, 1)...)
		}
		last := head && i == len(ids)-1
		if last {
			ops = append(ops, disp(g.normFin, g.fin, 1)...)
			for i := range g.chunks {
				c := &g.chunks[i]
				ops = append(ops, disp(c.lmP, c.lmG, c.lmX)...)
			}
		}
		var gets []rt.Get
		if last {
			gets = []rt.Get{{Buf: g.logits, N: m.Vocab * 4}}
		}
		out, err := g.dev.Run(ops, gets)
		if err != nil {
			return err
		}
		if last {
			lg := make([]float32, m.Vocab)
			for j := range lg {
				lg[j] = math.Float32frombits(binary.LittleEndian.Uint32(out[j*4:]))
			}
			d.lg = lg
		}
		d.pos++
	}
	return nil
}

// Dawn's storage binding limit. A larger embedding is several chunks.
const maxStorage = 2147483648

func chunkRows(rowBytes, vocab int) int {
	if rowBytes <= 0 || vocab <= 0 {
		return vocab
	}
	if vocab*rowBytes <= maxStorage {
		return vocab
	}
	n := maxStorage / rowBytes
	n -= n % 64
	if n < 64 {
		n = 64
	}
	return n
}

func (g *gpuRT) uploadEmbed(m *Model) error {
	rowBytes := m.D * 2
	if rowBytes <= 0 || len(m.embed) < m.Vocab*rowBytes {
		return fmt.Errorf("embedding is %d bytes, want %d", len(m.embed), m.Vocab*rowBytes)
	}
	step := chunkRows(rowBytes, m.Vocab)
	for start := 0; start < m.Vocab; {
		n := step
		if start+n > m.Vocab {
			n = m.Vocab - start
		}
		blob := m.embed[start*rowBytes : (start+n)*rowBytes]
		name := fmt.Sprintf("emb%d", start)
		if _, err := g.dev.Buf(name, blob, len(blob)); err != nil {
			return err
		}
		g.chunks = append(g.chunks, embChunk{start: start, rows: n})
		start += n
	}
	return nil
}

func (g *gpuRT) chunk(id int) (*embChunk, int, error) {
	for i := range g.chunks {
		c := &g.chunks[i]
		if id >= c.start && id < c.start+c.rows {
			return c, id - c.start, nil
		}
	}
	return nil, 0, fmt.Errorf("token %d is outside the embedding", id)
}

func disp(pipe, group, x uint32) []rt.Op {
	return []rt.Op{{Tag: rt.OpDispatch, Pipe: pipe, Group: group, X: x, Y: 1, Z: 1}}
}

func linOps(l linG) []rt.Op {
	ops := make([]rt.Op, len(l.pipe))
	for i := range l.pipe {
		ops[i] = rt.Op{Tag: rt.OpDispatch, Pipe: l.pipe[i], Group: l.grp[i], X: l.x, Y: 1, Z: 1}
	}
	return ops
}

func f32b(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}
