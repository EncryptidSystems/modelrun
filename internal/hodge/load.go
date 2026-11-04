// Package hodge runs a Hodge-Kuramoto checkpoint. A packed linear is
// hodgelinear.wgsl, every digit limb included. A dense bf16 linear is
// dense.wgsl on the float activation. The embedding, norms, rope, and
// attention follow those kernels. config.json's model_type stamp is not
// the runtime name.
package hodge

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"sync"

	"hodgemix/bnz4"
)

type got struct {
	b     []byte
	shape []int
}

var (
	layerRE = regexp.MustCompile(`^model\.layers\.(\d+)\.(.+)$`)
	embedRE = regexp.MustCompile(`^model\.embed_tokens\.weight\[(\d+):(\d+)\]$`)
)

func logf(f string, a ...any) {
	if logger != nil {
		logger(fmt.Sprintf(f, a...))
	}
}

// logger is set by the CLI. The TUI leaves it nil.
var logger func(string)

// SetLog receives load and token progress lines.
func SetLog(fn func(string)) { logger = fn }

func openBNZ(path string) (*os.File, *bnz4.Index, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	hdr := make([]byte, 12)
	if _, err := io.ReadFull(f, hdr); err != nil {
		f.Close()
		return nil, nil, 0, err
	}
	if string(hdr[:4]) != bnz4.Magic {
		f.Close()
		return nil, nil, 0, fmt.Errorf("%s is not a BNZ4 checkpoint", path)
	}
	n := int64(binary.LittleEndian.Uint64(hdr[4:]))
	buf := make([]byte, 12+n)
	copy(buf, hdr)
	if _, err := io.ReadFull(f, buf[12:]); err != nil {
		f.Close()
		return nil, nil, 0, err
	}
	ix, base, err := bnz4.ReadIndex(buf)
	if err != nil {
		f.Close()
		return nil, nil, 0, err
	}
	return f, ix, base, nil
}

func keep(name string) bool {
	switch name {
	case "", "gap", "header", "tail":
		return false
	}
	if len(name) >= 7 && name[:7] == "bnzmix." {
		return false
	}
	return true
}

// decodeAll reads a BNZ4 checkpoint, or the safetensors file that checkpoint
// is compressed from. Float tensors arrive as bf16 bytes. Packed U8 weights
// and I32 radix values arrive as their raw bytes.
func decodeAll(path string) (map[string]got, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var mag [4]byte
	_, err = io.ReadFull(f, mag[:])
	f.Close()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if string(mag[:]) == bnz4.Magic {
		return decodeBNZ(path)
	}
	return decodeST(path)
}

func decodeBNZ(path string) (map[string]got, error) {
	f, ix, base, err := openBNZ(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var want []int
	for i := range ix.Segments {
		if keep(ix.Segments[i].Name) {
			want = append(want, i)
		}
	}
	logf("decoding %d tensors from %s", len(want), filepath.Base(path))
	out := make(map[string]got, len(want))
	var mu sync.Mutex
	var first error
	jobs := make(chan int)
	var wg sync.WaitGroup
	n := runtime.NumCPU()
	if n < 1 {
		n = 1
	}
	for w := 0; w < n; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				sg := ix.Segments[i]
				c := make([]byte, sg.Len)
				nread, err := f.ReadAt(c, base+sg.Off)
				if err == nil && nread != len(c) {
					err = fmt.Errorf("short read %d", nread)
				}
				var raw []byte
				if err == nil {
					raw, err = bnz4.DecodeSeg(&sg, c)
				}
				mu.Lock()
				if err != nil && first == nil {
					first = fmt.Errorf("%s: %w", sg.Name, err)
				} else if err == nil {
					out[sg.Name] = got{b: raw, shape: append([]int(nil), sg.Shape...)}
				}
				mu.Unlock()
			}
		}()
	}
	for _, i := range want {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if first != nil {
		return nil, first
	}
	logf("decoded %d tensors", len(out))
	return out, nil
}

type stItem struct {
	name  string
	dtype string
	shape []int
	off   int64
	n     int
}

func decodeST(path string) (map[string]got, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() < 8 {
		return nil, fmt.Errorf("%s is not a safetensors checkpoint", path)
	}
	var nb [8]byte
	if _, err := io.ReadFull(f, nb[:]); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	n64 := binary.LittleEndian.Uint64(nb[:])
	if n64 == 0 || n64 > uint64(st.Size()-8) {
		return nil, fmt.Errorf("%s is not a safetensors checkpoint", path)
	}
	hj := make([]byte, n64)
	if _, err := io.ReadFull(f, hj); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(hj, &hdr); err != nil {
		return nil, fmt.Errorf("%s is not a safetensors checkpoint", path)
	}
	base := int64(8 + n64)
	var want []stItem
	for name, raw := range hdr {
		if name == "__metadata__" || !keep(name) {
			continue
		}
		var info struct {
			Dtype string  `json:"dtype"`
			Shape []int   `json:"shape"`
			Off   []int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("%s: %s: %w", path, name, err)
		}
		if len(info.Off) != 2 || info.Off[0] < 0 || info.Off[1] < info.Off[0] {
			return nil, fmt.Errorf("%s: %s: bad data_offsets", path, name)
		}
		need, ok := stBytes(info.Dtype, info.Shape)
		if !ok {
			return nil, fmt.Errorf("%s: %s: dtype %q", path, name, info.Dtype)
		}
		if info.Off[1]-info.Off[0] != need {
			return nil, fmt.Errorf("%s: %s is %d bytes, %s %v is %d", path, name, info.Off[1]-info.Off[0], info.Dtype, info.Shape, need)
		}
		if base+info.Off[1] > st.Size() {
			return nil, fmt.Errorf("%s: %s runs past the end of the file", path, name)
		}
		if need > int64(^uint(0)>>1) {
			return nil, fmt.Errorf("%s: %s is too large", path, name)
		}
		want = append(want, stItem{
			name: name, dtype: info.Dtype, shape: append([]int(nil), info.Shape...),
			off: base + info.Off[0], n: int(need),
		})
	}
	if len(want) == 0 {
		return nil, fmt.Errorf("%s: safetensors has no tensors", path)
	}
	logf("reading %d tensors from %s", len(want), filepath.Base(path))
	out := make(map[string]got, len(want))
	var mu sync.Mutex
	var first error
	jobs := make(chan stItem)
	var wg sync.WaitGroup
	nw := runtime.NumCPU()
	if nw < 1 {
		nw = 1
	}
	for w := 0; w < nw; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for it := range jobs {
				c := make([]byte, it.n)
				var err error
				if it.n > 0 {
					var nread int
					nread, err = f.ReadAt(c, it.off)
					if err == nil && nread != len(c) {
						err = fmt.Errorf("short read %d", nread)
					}
				}
				var raw []byte
				if err == nil {
					raw, err = asBF16(it.dtype, c)
				}
				mu.Lock()
				if err != nil && first == nil {
					first = fmt.Errorf("%s: %w", it.name, err)
				} else if err == nil {
					out[it.name] = got{b: raw, shape: it.shape}
				}
				mu.Unlock()
			}
		}()
	}
	for _, it := range want {
		jobs <- it
	}
	close(jobs)
	wg.Wait()
	if first != nil {
		return nil, first
	}
	logf("read %d tensors", len(out))
	return out, nil
}

func stElem(dtype string) (int64, bool) {
	switch dtype {
	case "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0":
		return 1, true
	case "F16", "BF16", "I16", "U16":
		return 2, true
	case "F32", "I32", "U32":
		return 4, true
	case "F64", "I64", "U64":
		return 8, true
	default:
		return 0, false
	}
}

func stBytes(dtype string, shape []int) (int64, bool) {
	es, ok := stElem(dtype)
	if !ok {
		return 0, false
	}
	n := int64(1)
	for _, d := range shape {
		if d < 0 {
			return 0, false
		}
		if d > 0 && n > (1<<62)/int64(d) {
			return 0, false
		}
		n *= int64(d)
	}
	if n > 0 && n > (1<<62)/es {
		return 0, false
	}
	return n * es, true
}

// Dawn reads bf16. The non-ternary tensors are stored as bf16, f16, or f32.
func asBF16(dtype string, b []byte) ([]byte, error) {
	switch dtype {
	case "F16":
		return f16sToBF16(b)
	case "F32":
		return f32sToBF16(b)
	default:
		return b, nil
	}
}

func f16to32(h uint16) float32 {
	sgn := uint32(h>>15) << 31
	e := int((h >> 10) & 31)
	m := uint32(h & 1023)
	switch {
	case e == 0:
		if m == 0 {
			return math.Float32frombits(sgn)
		}
		v := float32(m) / 1024 / 16384
		if sgn != 0 {
			v = -v
		}
		return v
	case e == 31:
		return math.Float32frombits(sgn | 0x7f800000 | m<<13)
	}
	return math.Float32frombits(sgn | uint32(e-15+127)<<23 | m<<13)
}

func f32tobf16(f float32) uint16 {
	b := math.Float32bits(f)
	return uint16((b + 0x7fff + ((b >> 16) & 1)) >> 16)
}

func f16sToBF16(b []byte) ([]byte, error) {
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("f16 length %d", len(b))
	}
	o := make([]byte, len(b))
	for i := 0; i < len(b); i += 2 {
		binary.LittleEndian.PutUint16(o[i:], f32tobf16(f16to32(binary.LittleEndian.Uint16(b[i:]))))
	}
	return o, nil
}

func f32sToBF16(b []byte) ([]byte, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("f32 length %d", len(b))
	}
	o := make([]byte, len(b)/2)
	for i := 0; i < len(o); i += 2 {
		f := math.Float32frombits(binary.LittleEndian.Uint32(b[2*i:]))
		binary.LittleEndian.PutUint16(o[i:], f32tobf16(f))
	}
	return o, nil
}

func bf16s(b []byte) ([]float32, error) {
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("bf16 length %d", len(b))
	}
	o := make([]float32, len(b)/2)
	for i := range o {
		o[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
	}
	return o, nil
}

func bf16one(b []byte) (float32, error) {
	if len(b) < 2 {
		return 0, fmt.Errorf("scale is %d bytes", len(b))
	}
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[:2])) << 16), nil
}

func unit(w []float32) bool {
	if len(w) == 0 {
		return false
	}
	for _, v := range w {
		if v != 1 {
			return false
		}
	}
	return true
}

type limb struct {
	w      []byte
	s      float32
	center int
}

type lin struct {
	n, k  int
	w     []byte // packed U8, nil when f is set
	s     float32
	limbs []limb
	f     []byte // dense bf16 [n, k], nil when packed
	bias  []byte // optional dense bf16 bias, length n*2
}

func need(t map[string]got, name string) (got, error) {
	g, ok := t[name]
	if !ok {
		return got{}, fmt.Errorf("checkpoint has no %s", name)
	}
	return g, nil
}

// shape2 reports whether a tensor's shape is absent or exactly [r, c].
func shape2(shape []int, r, c int) bool {
	if len(shape) == 0 {
		return true
	}
	return len(shape) == 2 && shape[0] == r && shape[1] == c
}

// bf16Row is one dense row dotted with x. Pairs match lmhead.wgsl and dense.wgsl:
// the low 16 bits of each u32 are element 2k, the high 16 bits are element 2k+1.
func bf16Row(row []byte, x []float32) float32 {
	var a float32
	for k := 0; k < len(x)/2; k++ {
		w := binary.LittleEndian.Uint32(row[4*k:])
		a += x[2*k]*math.Float32frombits(w<<16) + x[2*k+1]*math.Float32frombits(w&0xffff0000)
	}
	return a
}

func buildLin(t map[string]got, prefix string, n, k int) (lin, error) {
	g, err := need(t, prefix+".weight")
	if err != nil {
		return lin{}, err
	}
	want := (n / 4) * k
	dense := n * k * 2
	if k%2 == 0 && len(g.b) == dense && shape2(g.shape, n, k) {
		l := lin{n: n, k: k, f: g.b}
		if b, ok := t[prefix+".bias"]; ok {
			if len(b.b) != n*2 {
				return lin{}, fmt.Errorf("%s.bias is %d bytes, want %d", prefix, len(b.b), n*2)
			}
			if (len(b.shape) == 1 && b.shape[0] != n) || len(b.shape) > 1 {
				return lin{}, fmt.Errorf("%s.bias shape %v, want [%d]", prefix, b.shape, n)
			}
			l.bias = b.b
		}
		return l, nil
	}
	if len(g.b) != want || !shape2(g.shape, n/4, k) {
		return lin{}, fmt.Errorf("%s.weight is %d bytes shape %v, want packed %d [%d %d] or bf16 %d [%d %d]",
			prefix, len(g.b), g.shape, want, n/4, k, dense, n, k)
	}
	sc, err := need(t, prefix+".weight_scale")
	if err != nil {
		return lin{}, err
	}
	s, err := bf16one(sc.b)
	if err != nil {
		return lin{}, fmt.Errorf("%s: %w", prefix, err)
	}
	l := lin{n: n, k: k, w: g.b, s: s}
	for i := 1; ; i++ {
		name := fmt.Sprintf("%s.weight_limb%d", prefix, i)
		lb, ok := t[name]
		if !ok {
			break
		}
		if len(lb.b) != want {
			return lin{}, fmt.Errorf("%s is %d bytes, want %d", name, len(lb.b), want)
		}
		ss, err := need(t, name+"_scale")
		if err != nil {
			return lin{}, err
		}
		sf, err := bf16one(ss.b)
		if err != nil {
			return lin{}, err
		}
		rb, err := need(t, name+"_radix")
		if err != nil {
			return lin{}, err
		}
		if len(rb.b) < 4 {
			return lin{}, fmt.Errorf("%s_radix is %d bytes", name, len(rb.b))
		}
		radix := binary.LittleEndian.Uint32(rb.b[:4])
		if radix < 2 {
			return lin{}, fmt.Errorf("%s radix %d", name, radix)
		}
		l.limbs = append(l.limbs, limb{w: lb.b, s: sf, center: int((radix - 1) / 2)})
	}
	return l, nil
}

func vec(t map[string]got, name string, n int, optional bool) ([]float32, error) {
	g, ok := t[name]
	if !ok {
		if optional {
			return nil, nil
		}
		return nil, fmt.Errorf("checkpoint has no %s", name)
	}
	w, err := bf16s(g.b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if n > 0 && len(w) != n {
		return nil, fmt.Errorf("%s has %d values, want %d", name, len(w), n)
	}
	if unit(w) && optional {
		return nil, nil
	}
	return w, nil
}

type layer struct {
	attnN, ffnN []float32
	attnS, ffnS []float32
	qn, kn      []float32
	q, k, v, o  lin
	gate, up    lin
	down        lin
}

func assembleEmbed(t map[string]got, vocab, d int) ([]byte, error) {
	// Safetensors stores the embedding as one tensor. BNZ4 splits that matrix
	// into row ranges so each range can be scheduled on its own.
	wantAll := vocab * d * 2
	if g, ok := t["model.embed_tokens.weight"]; ok {
		if len(g.b) != wantAll {
			return nil, fmt.Errorf("model.embed_tokens.weight is %d bytes, want %d", len(g.b), wantAll)
		}
		return g.b, nil
	}
	emb := make([]byte, wantAll)
	filled := 0
	for name, g := range t {
		m := embedRE.FindStringSubmatch(name)
		if m == nil {
			continue
		}
		r0, _ := strconv.Atoi(m[1])
		r1, _ := strconv.Atoi(m[2])
		if r1 > vocab || r0 < 0 || r1 < r0 {
			return nil, fmt.Errorf("%s is outside vocab %d", name, vocab)
		}
		want := (r1 - r0) * d * 2
		if len(g.b) != want {
			return nil, fmt.Errorf("%s is %d bytes, want %d", name, len(g.b), want)
		}
		copy(emb[r0*d*2:], g.b)
		filled += want
	}
	if filled != len(emb) {
		return nil, fmt.Errorf("embedding covered %d bytes, want %d", filled, len(emb))
	}
	return emb, nil
}

func layerCount(t map[string]got) int {
	n := 0
	for name := range t {
		m := layerRE.FindStringSubmatch(name)
		if m == nil {
			continue
		}
		i, _ := strconv.Atoi(m[1])
		if i+1 > n {
			n = i + 1
		}
	}
	return n
}
