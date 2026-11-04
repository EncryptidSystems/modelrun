package synthid

import "testing"

func TestThreefryVector(t *testing.T) {
	// 654 is an arbitrary input to the hash, not a watermark key.
	o1, o2 := threefry(0, 654, 0, 0)
	if o1 != 1129024486 || o2 != 1249734243 {
		t.Fatalf("threefry %d %d", o1, o2)
	}
	k1, k2 := seed(654)
	c1, c2 := foldAll(k1, k2, []int64{0, 0, 0, 0})
	if c1 != 4089424118 || c2 != 1132592832 {
		t.Fatalf("context %d %d", c1, c2)
	}
	bits := gBits(c1, c2, 7, 30)
	var got string
	ones := 0
	for _, b := range bits {
		if b == 0 {
			got += "0"
		} else {
			got += "1"
			ones++
		}
	}
	const want = "010101011100111111011010111110"
	if got != want || ones != 20 {
		t.Fatalf("g %s ones %d", got, ones)
	}
}

func TestBinomSF(t *testing.T) {
	if binomSF(0, 0) != 1 {
		t.Fatal("empty")
	}
	// Binomial(2, 1/2): P(X >= 2) = 0.25
	if g := binomSF(2, 2); g < 0.249 || g > 0.251 {
		t.Fatalf("sf %v", g)
	}
}

func TestExpectedMean(t *testing.T) {
	g, err := ExpectedMeanG(4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if g != 0.5+0.25*(1-0.25) {
		t.Fatal(g)
	}
}
