package mark

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// The 18-cell key. A cell is in it when every poem's centroid is on the same
// side of the vocabulary mean and at least 2 standard deviations from a
// random vocabulary centroid.
var keyCells = []int{0, 2, 4, 5, 9, 10, 20, 26, 28, 29, 30, 46, 47, 50, 53, 59, 61, 63}
var keySign = []float64{1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1}

func TestKeyExtracts(t *testing.T) {
	model := `C:\Users\grego\models\dev\hodge-3.5-instruct`
	dir := `C:\Users\grego\watermarks`
	if _, err := os.Stat(filepath.Join(model, "model.cochain.safetensors")); err != nil {
		t.Skip(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "poem.txt")); err != nil {
		t.Skip(err)
	}
	c, err := Open(filepath.Join(model, "tokenizer.json"), filepath.Join(model, "tokenizer_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"poem-1.txt", "poem-2.txt", "poem.txt"}
	ch := make([]Chain, len(names))
	for i, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		ch[i] = c.Chain(name, string(b))
	}
	C := len(ch[0].Centroid)
	rng := rand.New(rand.NewSource(11))
	const trials = 64
	null := make([][][]float64, len(ch))
	for i := range ch {
		null[i] = make([][]float64, trials)
		for k := 0; k < trials; k++ {
			null[i][k] = centeredMean(c, len(ch[i].Anchors), rng)
		}
	}
	z := cellZ(ch, null)

	// The published key has to be exactly the cells the rule selects.
	got := map[int]float64{}
	for g := 0; g < C; g++ {
		s := keyOf(ch, g)
		if s == 0 {
			continue
		}
		ok := true
		for i := range ch {
			if math.Abs(z[i][g]) < 2 || sign1(z[i][g]) != s {
				ok = false
				break
			}
		}
		if ok {
			got[g] = s
		}
	}
	var extra, missing []int
	claimed := map[int]float64{}
	for i, g := range keyCells {
		claimed[g] = keySign[i]
	}
	for g, s := range got {
		if claimed[g] != s {
			extra = append(extra, g)
		}
	}
	for g := range claimed {
		if _, ok := got[g]; !ok {
			missing = append(missing, g)
		}
	}
	t.Logf("rule selected %d  extra %v  missing %v", len(got), extra, missing)
	for g := 0; g < C; g++ {
		if keyOf(ch, g) == 0 {
			continue
		}
		m := math.Abs(z[0][g])
		for i := 1; i < len(ch); i++ {
			if a := math.Abs(z[i][g]); a < m {
				m = a
			}
		}
		in := " "
		if _, ok := got[g]; ok {
			in = "*"
		}
		t.Logf("%s c%-2d  min|z| %5.2f  z %6.2f %6.2f %6.2f", in, g, m, z[0][g], z[1][g], z[2][g])
	}
	for i, g := range keyCells {
		if got[g] != keySign[i] {
			t.Errorf("cell %d sign %.0f, key says %.0f", g, got[g], keySign[i])
		}
	}

	// Each poem, read alone, must reproduce every sign.
	for i := range ch {
		h := keyHits(ch[i].Centroid)
		t.Logf("extract %s  %d/%d", ch[i].Name, h, len(keyCells))
		if h != len(keyCells) {
			t.Errorf("%s reproduced %d of %d key signs", ch[i].Name, h, len(keyCells))
		}
	}

	// Random vocabulary of the same length.
	for i := range ch {
		hi, sum := 0, 0
		for _, r := range null[i] {
			h := keyHits(r)
			sum += h
			if h > hi {
				hi = h
			}
		}
		t.Logf("random %s  mean %.1f/%d  max %d/%d", ch[i].Name, float64(sum)/float64(trials), len(keyCells), hi, len(keyCells))
		if hi >= len(keyCells) {
			t.Errorf("a random text matched the whole key (%d/%d)", hi, len(keyCells))
		}
	}

	// Discover the cells from two poems. Read the third.
	for hold := range ch {
		var cells []int
		var signs []float64
		for g := 0; g < C; g++ {
			a, b := (hold+1)%3, (hold+2)%3
			if sign1(ch[a].Centroid[g]) != sign1(ch[b].Centroid[g]) {
				continue
			}
			if math.Abs(z[a][g]) < 2 || math.Abs(z[b][g]) < 2 {
				continue
			}
			if sign1(z[a][g]) != sign1(ch[a].Centroid[g]) || sign1(z[b][g]) != sign1(ch[b].Centroid[g]) {
				continue
			}
			cells = append(cells, g)
			signs = append(signs, sign1(ch[a].Centroid[g]))
		}
		h := hitsAt(ch[hold].Centroid, cells, signs)
		rmax := 0
		var missed []int
		for i, g := range cells {
			if sign1(ch[hold].Centroid[g]) != signs[i] {
				missed = append(missed, g)
			}
		}
		for _, r := range null[hold] {
			if n := hitsAt(r, cells, signs); n > rmax {
				rmax = n
			}
		}
		t.Logf("held out %s  discovered %d cells  poem %d/%d  missed %v  random max %d/%d",
			ch[hold].Name, len(cells), h, len(cells), missed, rmax, len(cells))
	}

	// Cells whose smallest |z| stays clear of the cutoff.
	var coreC []int
	var coreS []float64
	for g := 0; g < C; g++ {
		s := keyOf(ch, g)
		if s == 0 {
			continue
		}
		m := math.Abs(z[0][g])
		for i := 1; i < len(ch); i++ {
			if a := math.Abs(z[i][g]); a < m {
				m = a
			}
		}
		if m >= 4 {
			coreC = append(coreC, g)
			coreS = append(coreS, s)
		}
	}
	for i := range ch {
		h := hitsAt(ch[i].Centroid, coreC, coreS)
		hi := 0
		for _, r := range null[i] {
			if n := hitsAt(r, coreC, coreS); n > hi {
				hi = n
			}
		}
		t.Logf("core %s  cells %v  poem %d/%d  random max %d/%d", ch[i].Name, coreC, h, len(coreC), hi, len(coreC))
	}

	// How many leading words the read needs.
	for _, n := range []int{16, 24, 32, 48, 64, 96} {
		for i := range ch {
			if len(ch[i].Anchors) < n {
				continue
			}
			h := keyHits(meanOf(ch[i].Anchors[:n]))
			t.Logf("prefix %s  %d words  %d/%d", ch[i].Name, n, h, len(keyCells))
		}
	}
}

func centeredMean(c *Checker, n int, rng *rand.Rand) []float64 {
	row := make([]float64, c.C)
	for _, w := range c.randRows(n, rng) {
		for g := range row {
			row[g] += w[g]
		}
	}
	k := float64(n)
	for g := range row {
		row[g] = row[g]/k - c.mu[g]
	}
	return row
}

func cellZ(ch []Chain, null [][][]float64) [][]float64 {
	C := len(ch[0].Centroid)
	z := make([][]float64, len(ch))
	for i := range ch {
		z[i] = make([]float64, C)
		for g := 0; g < C; g++ {
			var m float64
			for _, r := range null[i] {
				m += r[g]
			}
			m /= float64(len(null[i]))
			var s float64
			for _, r := range null[i] {
				d := r[g] - m
				s += d * d
			}
			sd := math.Sqrt(s / float64(len(null[i])))
			if sd < 1e-9 {
				sd = 1
			}
			z[i][g] = (ch[i].Centroid[g] - m) / sd
		}
	}
	return z
}

func keyOf(ch []Chain, g int) float64 {
	s := sign1(ch[0].Centroid[g])
	for i := 1; i < len(ch); i++ {
		if sign1(ch[i].Centroid[g]) != s {
			return 0
		}
	}
	return s
}

func keyHits(v []float64) int {
	return hitsAt(v, keyCells, keySign)
}

func hitsAt(v []float64, cells []int, signs []float64) int {
	n := 0
	for i, g := range cells {
		if sign1(v[g]) == signs[i] {
			n++
		}
	}
	return n
}

func meanOf(xs [][]float64) []float64 {
	o := make([]float64, len(xs[0]))
	k := float64(len(xs))
	for _, v := range xs {
		for i := range o {
			o[i] += v[i] / k
		}
	}
	return o
}

func sign1(v float64) float64 {
	if v >= 0 {
		return 1
	}
	return -1
}
