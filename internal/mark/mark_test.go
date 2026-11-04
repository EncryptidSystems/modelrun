package mark

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenMissing(t *testing.T) {
	c, err := Open(filepath.Join(t.TempDir(), "tokenizer.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	if c != nil {
		t.Fatal("missing cochain opened")
	}
}

func TestSharedDirectionClearsRandom(t *testing.T) {
	const C = 8
	const V = 300
	co := make([]int32, V*C)
	rng := rand.New(rand.NewSource(2))
	for i := range co {
		co[i] = int32(rng.Intn(5) - 2)
	}
	for id := 0; id < 60; id++ {
		co[id*C] = 50
	}
	mu := make([]float64, C)
	for id := 0; id < V; id++ {
		for g := 0; g < C; g++ {
			mu[g] += float64(co[id*C+g])
		}
	}
	for g := range mu {
		mu[g] /= float64(V)
	}
	marked := make([][]float64, 60)
	for id := range marked {
		row := make([]float64, C)
		for g := 0; g < C; g++ {
			row[g] = float64(co[id*C+g])
		}
		marked[id] = row
	}
	al := alignment(marked, mu)
	c := &Checker{co: co, V: V, C: C, mu: mu, nulls: map[int]nullFit{}}
	for id := 60; id < V; id++ {
		c.pool = append(c.pool, id)
	}
	fit := c.bar(len(marked))
	if al <= fit.alMax {
		t.Fatalf("alignment %.3f did not clear null max %.3f", al, fit.alMax)
	}
}

func TestCellShareMatchesDFT(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const R, C = 6, 8
	raw := make([][]float64, R)
	mu := make([]float64, C)
	for r := 0; r < R; r++ {
		raw[r] = make([]float64, C)
		for c := 0; c < C; c++ {
			raw[r][c] = rng.NormFloat64()
		}
	}
	got := cellShare(raw, mu)
	s := splitHodge(raw)
	want := s.gCell / s.tot
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("cell share %.12f  dft %.12f", got, want)
	}
}

func TestPoems(t *testing.T) {
	model := `C:\Users\grego\models\dev\hodge-3.5-instruct`
	dir := `C:\Users\grego\watermarks`
	if _, err := os.Stat(filepath.Join(model, "model.cochain.safetensors")); err != nil {
		t.Skip(err)
	}
	c, err := Open(filepath.Join(model, "tokenizer.json"), filepath.Join(model, "tokenizer_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if c == nil {
		t.Fatal("cochain not opened")
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Skip(err)
	}
	n := 0
	for _, e := range ents {
		if e.IsDir() || filepath.Ext(e.Name()) != ".txt" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		s := c.Score(string(b))
		t.Logf("%s  words %d  alignment %.3f  null %.3f  cell %.4f  cell_null %.4f  marked %v",
			e.Name(), s.Words, s.Alignment, s.NullMax, s.CellShare, s.CellNull, s.Marked)
		if s.Words < 100 {
			t.Fatalf("%s kept %d words", e.Name(), s.Words)
		}
		if !s.Marked {
			t.Fatalf("%s cell share %.4f did not clear null %.4f", e.Name(), s.CellShare, s.CellNull)
		}
		if err := c.Refuse(string(b)); !errors.Is(err, ErrMarked) {
			t.Fatalf("%s refuse: %v", e.Name(), err)
		}
		n++
	}
	if n < 3 {
		t.Fatalf("poems %d", n)
	}
}
