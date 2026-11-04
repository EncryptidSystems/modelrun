package mark

import (
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// Hodge bins as in bnz_entropy_stream.hodge_energy: gradient is the zero
// row or column of the DFT, harmonic is the four corner bins, curl is the rest.
func TestHodgeStructure(t *testing.T) {
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
	rng := rand.New(rand.NewSource(3))
	poems := make([]hodgeSplit, len(ch))
	for i := range ch {
		poems[i] = splitHodge(ch[i].Anchors)
		t.Logf("%s  words %d  G %.0f  curl %.0f  H %.0f  shares g %.4f c %.4f h %.4f  cell %.4f  word %.4f",
			ch[i].Name, len(ch[i].Anchors), poems[i].g, poems[i].c, poems[i].h,
			poems[i].g/poems[i].tot, poems[i].c/poems[i].tot, poems[i].h/poems[i].tot,
			poems[i].gCell/poems[i].tot, poems[i].gWord/poems[i].tot)
		t.Logf("  |harmonic bins| %.0f %.0f %.0f %.0f",
			cmplx.Abs(poems[i].harm[0]), cmplx.Abs(poems[i].harm[1]),
			cmplx.Abs(poems[i].harm[2]), cmplx.Abs(poems[i].harm[3]))
	}
	for i := 0; i < 4; i++ {
		n := len(ch[i%len(ch)].Anchors)
		rows := c.randRows(n, rng)
		for r := range rows {
			for g := range rows[r] {
				rows[r][g] -= c.mu[g]
			}
		}
		s := splitHodge(rows)
		t.Logf("random n=%d  shares g %.4f c %.4f h %.4f  cell %.4f  word %.4f",
			n, s.g/s.tot, s.c/s.tot, s.h/s.tot, s.gCell/s.tot, s.gWord/s.tot)
	}
	for i := 0; i < len(poems); i++ {
		for j := i + 1; j < len(poems); j++ {
			t.Logf("%s · %s  harmonic magnitude cosine %.3f  phase-invariant overlap %.3f",
				ch[i].Name, ch[j].Name, magCos(poems[i].harm, poems[j].harm), phaseAlign(poems[i].harm, poems[j].harm))
		}
	}
}

type hodgeSplit struct {
	g, c, h, tot float64
	gCell        float64 // kr = 0, the centroid's cell spectrum
	gWord        float64 // kc = 0, kr > 0, per-word mass
	harm         [4]complex128
}

func splitHodge(M [][]float64) hodgeSplit {
	hat := dft2(M)
	R, C := len(M), len(M[0])
	var s hodgeSplit
	norm := float64(R * C)
	hi := 0
	for kr := 0; kr < R; kr++ {
		for kc := 0; kc < C; kc++ {
			z := hat[kr][kc]
			e := cmplx.Abs(z) * cmplx.Abs(z) / norm
			s.tot += e
			switch {
			case kr == 0 || kc == 0:
				s.g += e
				if kr == 0 && kc != 0 {
					s.gCell += e
				} else if kc == 0 && kr != 0 {
					s.gWord += e
				}
			case (kr == 1 || kr == R-1) && (kc == 1 || kc == C-1):
				s.h += e
				s.harm[hi] = z
				hi++
			default:
				s.c += e
			}
		}
	}
	return s
}

func dft2(M [][]float64) [][]complex128 {
	R, C := len(M), len(M[0])
	row := make([][]complex128, R)
	for r := 0; r < R; r++ {
		row[r] = make([]complex128, C)
		for kc := 0; kc < C; kc++ {
			var re, im float64
			for c := 0; c < C; c++ {
				ang := -2 * math.Pi * float64(kc*c) / float64(C)
				re += M[r][c] * math.Cos(ang)
				im += M[r][c] * math.Sin(ang)
			}
			row[r][kc] = complex(re, im)
		}
	}
	hat := make([][]complex128, R)
	for kr := 0; kr < R; kr++ {
		hat[kr] = make([]complex128, C)
		wr := -2 * math.Pi / float64(R)
		for kc := 0; kc < C; kc++ {
			var re, im float64
			for r := 0; r < R; r++ {
				ang := wr * float64(kr*r)
				cs, sn := math.Cos(ang), math.Sin(ang)
				re += real(row[r][kc])*cs - imag(row[r][kc])*sn
				im += real(row[r][kc])*sn + imag(row[r][kc])*cs
			}
			hat[kr][kc] = complex(re, im)
		}
	}
	return hat
}

func magCos(a, b [4]complex128) float64 {
	var d, na, nb float64
	for i := range a {
		x, y := cmplx.Abs(a[i]), cmplx.Abs(b[i])
		d += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return d / math.Sqrt(na*nb)
}

func phaseAlign(a, b [4]complex128) float64 {
	var d, na, nb float64
	for i := range a {
		d += cmplx.Abs(a[i] * cmplx.Conj(b[i]))
		na += cmplx.Abs(a[i]) * cmplx.Abs(a[i])
		nb += cmplx.Abs(b[i]) * cmplx.Abs(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return d / math.Sqrt(na*nb)
}


