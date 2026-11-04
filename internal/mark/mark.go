// Package mark scores one produced text against the model's integer cochain.
// The null is the same number of distinct vocabulary words. There is no corpus.
//
// The decision follows the frequentist shape of the public SynthID detector:
// one text, a statistic with a vocabulary null, and a repeated context left
// out of the count. The statistic is the cell-gradient share of cochain
// energy. SynthID's Threefry key and tournament are a different mechanism
// and are not applied here.
package mark

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode"

	"hodgemix/eval"
)

// Score is the watermark reading of one text.
// Alignment is how tightly its words share a direction after the vocabulary
// mean is removed. CellShare is the fraction of cochain energy in the
// cell-spectrum gradient, the part that sat above the vocabulary null.
// A repeated word is masked and does not enter either number.
// Marked is set once the text is long enough and CellShare clears the null.
type Score struct {
	Words     int     `json:"words"`
	Cells     int     `json:"cells"`
	Alignment float64 `json:"alignment"`
	Null      float64 `json:"null"`
	NullMax   float64 `json:"null_max"`
	CellShare float64 `json:"cell_share"`
	CellNull  float64 `json:"cell_null"`
	Marked    bool    `json:"marked"`
}

// Checker is one model cochain and the tokenizer that indexed it.
type Checker struct {
	co   []int32
	V, C int
	mu   []float64
	pool []int
	sess *eval.Session

	nullMu sync.Mutex
	nulls  map[int]nullFit
}

// nullFit is the vocabulary null for one word count.
// The first two fields are the alignment mean and max. The last two are the
// cell-gradient share mean and max.
type nullFit struct {
	alMean, alMax     float64
	cellMean, cellMax float64
}

var (
	cacheMu sync.Mutex
	cache   = map[string]*Checker{}
)

// Cached returns the checker for this tokenizer. A missing cochain is cached
// as a nil checker.
func Cached(tokenizer, tokcfg string) (*Checker, error) {
	key := tokenizer + "\x00" + tokcfg
	cacheMu.Lock()
	if c, ok := cache[key]; ok {
		cacheMu.Unlock()
		return c, nil
	}
	cacheMu.Unlock()
	c, err := Open(tokenizer, tokcfg)
	if err != nil {
		return nil, err
	}
	cacheMu.Lock()
	cache[key] = c
	cacheMu.Unlock()
	return c, nil
}

// Open reads model.cochain.safetensors beside the tokenizer. A missing file
// is not an error: the model has no cochain to score.
func Open(tokenizer, tokcfg string) (*Checker, error) {
	if tokenizer == "" {
		return nil, nil
	}
	path := filepath.Join(filepath.Dir(tokenizer), "model.cochain.safetensors")
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return openAt(path, tokenizer, tokcfg)
}

// OpenAt loads the cochain file and tokenizer you pass. It does not search
// for a safetensors name.
func OpenAt(cochain, tokenizer, tokcfg string) (*Checker, error) {
	if cochain == "" || tokenizer == "" {
		return nil, fmt.Errorf("cochain and tokenizer paths are required")
	}
	return openAt(cochain, tokenizer, tokcfg)
}

func openAt(path, tokenizer, tokcfg string) (*Checker, error) {
	co, V, C, err := loadCochain(path)
	if err != nil {
		return nil, err
	}
	sess, err := eval.OpenTok(tokenizer, tokcfg)
	if err != nil {
		return nil, err
	}
	mu := make([]float64, C)
	for id := 0; id < V; id++ {
		base := id * C
		for g := 0; g < C; g++ {
			mu[g] += float64(co[base+g])
		}
	}
	for g := range mu {
		mu[g] /= float64(V)
	}
	var pool []int
	for id := 0; id < V; id++ {
		if letterWord(sess.Decode([]int{id})) {
			pool = append(pool, id)
		}
	}
	if len(pool) < 8 {
		return nil, fmt.Errorf("%s: vocabulary has %d letter words", path, len(pool))
	}
	return &Checker{co: co, V: V, C: C, mu: mu, pool: pool, sess: sess, nulls: map[int]nullFit{}}, nil
}

// Chain is one text on the cochain. Anchors are word rows with the vocabulary
// mean removed. Centroid is the mean of those rows, the fingerprint candidate.
type Chain struct {
	Name      string
	Words     []string
	Anchors   [][]float64
	Centroid  []float64
	Alignment float64
}

// Chain places text on the cochain. Letter pieces only, in order.
func (c *Checker) Chain(name, text string) Chain {
	ch := Chain{Name: name}
	if c == nil || strings.TrimSpace(text) == "" {
		return ch
	}
	var raw [][]float64
	for _, ids := range c.sess.Segments(text) {
		if !letterWord(c.sess.Decode(ids)) {
			continue
		}
		row := make([]float64, c.C)
		for _, id := range ids {
			if id < 0 || id >= c.V {
				continue
			}
			base := id * c.C
			for g := 0; g < c.C; g++ {
				row[g] += float64(c.co[base+g])
			}
		}
		raw = append(raw, row)
		w := strings.TrimSpace(c.sess.Decode(ids))
		ch.Words = append(ch.Words, w)
	}
	if len(raw) == 0 {
		return ch
	}
	ch.Alignment = alignment(raw, c.mu)
	ch.Anchors = make([][]float64, len(raw))
	ch.Centroid = make([]float64, c.C)
	for i, row := range raw {
		cent := make([]float64, c.C)
		for g := 0; g < c.C; g++ {
			cent[g] = row[g] - c.mu[g]
			ch.Centroid[g] += cent[g]
		}
		ch.Anchors[i] = cent
	}
	k := float64(len(raw))
	for g := range ch.Centroid {
		ch.Centroid[g] /= k
	}
	return ch
}

// ErrMarked is returned when text carries the cochain watermark.
// The model must not read that text.
var ErrMarked = errors.New("watermarked input")

// Refuse reports ErrMarked when Score would mark text.
func (c *Checker) Refuse(text string) error {
	if c == nil {
		return nil
	}
	s := c.Score(text)
	if !s.Marked {
		return nil
	}
	return fmt.Errorf("%w: %d words, cell share %.4f above %.4f", ErrMarked, s.Words, s.CellShare, s.CellNull)
}

// Score places text on the cochain. Words are the tokenizer's letter pieces.
// A word already scored in this text is masked, the same role as SynthID's
// repeated-context mask: a repeat must not move the statistic.
func (c *Checker) Score(text string) Score {
	if c == nil || strings.TrimSpace(text) == "" {
		return Score{}
	}
	var keep [][]int
	seen := map[string]bool{}
	for _, ids := range c.sess.Segments(text) {
		w := strings.TrimSpace(c.sess.Decode(ids))
		if !letterWord(w) {
			continue
		}
		k := strings.ToLower(w)
		if seen[k] {
			continue
		}
		seen[k] = true
		keep = append(keep, ids)
	}
	if len(keep) == 0 {
		return Score{Cells: c.C}
	}
	xs := make([][]float64, len(keep))
	for i, ids := range keep {
		row := make([]float64, c.C)
		for _, id := range ids {
			if id < 0 || id >= c.V {
				continue
			}
			base := id * c.C
			for g := 0; g < c.C; g++ {
				row[g] += float64(c.co[base+g])
			}
		}
		xs[i] = row
	}
	al := alignment(xs, c.mu)
	share := cellShare(xs, c.mu)
	fit := c.bar(len(xs))
	return Score{
		Words: len(xs), Cells: c.C,
		Alignment: al, Null: fit.alMean, NullMax: fit.alMax,
		CellShare: share, CellNull: fit.cellMax,
		Marked: len(xs) >= 24 && share > fit.cellMax,
	}
}

func (c *Checker) bar(n int) nullFit {
	c.nullMu.Lock()
	defer c.nullMu.Unlock()
	if v, ok := c.nulls[n]; ok {
		return v
	}
	rng := rand.New(rand.NewSource(int64(n) + 1))
	const trials = 16
	var alSum, alMax, cellSum, cellMax float64
	for t := 0; t < trials; t++ {
		rows := c.randRows(n, rng)
		al := alignment(rows, c.mu)
		sh := cellShare(rows, c.mu)
		alSum += al
		cellSum += sh
		if t == 0 || al > alMax {
			alMax = al
		}
		if t == 0 || sh > cellMax {
			cellMax = sh
		}
	}
	fit := nullFit{alMean: alSum / trials, alMax: alMax, cellMean: cellSum / trials, cellMax: cellMax}
	c.nulls[n] = fit
	return fit
}

// cellShare is the cell-spectrum gradient (DFT row kr = 0, columns kc ≠ 0)
// as a fraction of cochain energy. It is the one-text statistic. A sign key
// is not estimated from the text.
func cellShare(xs [][]float64, mu []float64) float64 {
	if len(xs) == 0 || len(mu) == 0 {
		return 0
	}
	cent := make([]float64, len(mu))
	var tot float64
	for _, v := range xs {
		var n2 float64
		for i := range cent {
			d := v[i] - mu[i]
			cent[i] += d
			n2 += d * d
		}
		tot += n2
	}
	if tot == 0 {
		return 0
	}
	r := float64(len(xs))
	for i := range cent {
		cent[i] /= r
	}
	var n2, sum float64
	for _, x := range cent {
		n2 += x * x
		sum += x
	}
	gCell := r * (n2 - sum*sum/float64(len(mu)))
	if gCell < 0 {
		gCell = 0
	}
	return gCell / tot
}

func (c *Checker) randRows(n int, rng *rand.Rand) [][]float64 {
	if n > len(c.pool) {
		n = len(c.pool)
	}
	seen := map[int]bool{}
	out := make([][]float64, 0, n)
	for len(out) < n {
		id := c.pool[rng.Intn(len(c.pool))]
		if seen[id] {
			continue
		}
		seen[id] = true
		row := make([]float64, c.C)
		base := id * c.C
		for g := 0; g < c.C; g++ {
			row[g] = float64(c.co[base+g])
		}
		out = append(out, row)
	}
	return out
}

func alignment(xs [][]float64, mu []float64) float64 {
	cent := make([]float64, len(mu))
	for _, v := range xs {
		for i := range cent {
			cent[i] += v[i] - mu[i]
		}
	}
	k := float64(len(xs))
	for i := range cent {
		cent[i] /= k
	}
	var s float64
	buf := make([]float64, len(mu))
	for _, v := range xs {
		for i := range buf {
			buf[i] = v[i] - mu[i]
		}
		s += cosine(buf, cent)
	}
	return s / k
}

func cosine(a, b []float64) float64 {
	var d, na, nb float64
	for i := range a {
		d += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return d / math.Sqrt(na*nb)
}

func letterWord(s string) bool {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "'")
	if len([]rune(s)) < 2 {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

func loadCochain(path string) ([]int32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	var nb [8]byte
	if _, err := io.ReadFull(f, nb[:]); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", path, err)
	}
	n := binary.LittleEndian.Uint64(nb[:])
	if n == 0 || n > 1<<26 {
		return nil, 0, 0, fmt.Errorf("%s: header %d", path, n)
	}
	hb := make([]byte, n)
	if _, err := io.ReadFull(f, hb); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hb, &raw); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", path, err)
	}
	var meta map[string]string
	if m, ok := raw["__metadata__"]; ok {
		_ = json.Unmarshal(m, &meta)
	}
	if q := meta["quantum"]; q != "" && q != "1/4096" {
		return nil, 0, 0, fmt.Errorf("%s quantum %s, this tool reads 1/4096", path, q)
	}
	var info struct {
		Dtype string  `json:"dtype"`
		Shape []int   `json:"shape"`
		Off   []int64 `json:"data_offsets"`
	}
	body, ok := raw["cochain"]
	if !ok {
		return nil, 0, 0, fmt.Errorf("%s has no cochain", path)
	}
	if err := json.Unmarshal(body, &info); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", path, err)
	}
	if info.Dtype != "I32" || len(info.Shape) != 2 || len(info.Off) != 2 {
		return nil, 0, 0, fmt.Errorf("%s: cochain is %s %v", path, info.Dtype, info.Shape)
	}
	rows, cols := info.Shape[0], info.Shape[1]
	if rows < 1 || cols < 1 || info.Off[1]-info.Off[0] != int64(rows*cols*4) {
		return nil, 0, 0, fmt.Errorf("%s: cochain shape %v", path, info.Shape)
	}
	b := make([]byte, rows*cols*4)
	if _, err := f.ReadAt(b, 8+int64(n)+info.Off[0]); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", path, err)
	}
	co := make([]int32, rows*cols)
	for i := range co {
		co[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return co, rows, cols, nil
}
