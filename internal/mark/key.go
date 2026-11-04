package mark

import (
	"encoding/hex"
	"fmt"
)

// CorpusKey is the sign agreement of a corpus on the integer cochain.
// Key is an integer in [0, 2^63). Bit i is set when every text is positive
// on cell i. Cells 0 through 62 are in the integer. Cell 63 is reported in
// Cell63 and left out of Key. Code is the 2-bit agreement, cell 0 in the low
// bits of byte 0: 0 disagree, 1 shared plus, 2 shared minus. Mask is that
// code as text, cell 0 on the left.
type CorpusKey struct {
	Key    uint64
	Code   string
	Mask   string
	Agreed int
	Cells  int
	Cell63 int
	Texts  int
}

// KeyFrom extracts the corpus key. A corpus is at least two texts.
func KeyFrom(chains []Chain) (CorpusKey, error) {
	if len(chains) < 2 {
		return CorpusKey{}, fmt.Errorf("need at least two texts")
	}
	cells := len(chains[0].Centroid)
	if cells == 0 {
		return CorpusKey{}, fmt.Errorf("%s has no cochain", chains[0].Name)
	}
	for _, ch := range chains[1:] {
		if len(ch.Centroid) != cells {
			return CorpusKey{}, fmt.Errorf("%s has %d cells", ch.Name, len(ch.Centroid))
		}
	}
	code := make([]byte, cells)
	mask := make([]byte, cells)
	var key uint64
	agreed := 0
	cell63 := 0
	for g := 0; g < cells; g++ {
		sign := 1
		if chains[0].Centroid[g] < 0 {
			sign = -1
		}
		ok := true
		for i := 1; i < len(chains); i++ {
			s := 1
			if chains[i].Centroid[g] < 0 {
				s = -1
			}
			if s != sign {
				ok = false
				break
			}
		}
		if !ok {
			mask[g] = '.'
			continue
		}
		agreed++
		if sign > 0 {
			code[g] = 1
			mask[g] = '+'
			if g < 63 {
				key |= 1 << uint(g)
			} else if g == 63 {
				cell63 = 1
			}
		} else {
			code[g] = 2
			mask[g] = '-'
			if g == 63 {
				cell63 = -1
			}
		}
	}
	raw := make([]byte, (cells*2+7)/8)
	for g, v := range code {
		raw[g/4] |= v << uint((g%4)*2)
	}
	if key >= 1<<63 {
		return CorpusKey{}, fmt.Errorf("key does not fit in 63 bits")
	}
	return CorpusKey{
		Key: key, Code: hex.EncodeToString(raw), Mask: string(mask),
		Agreed: agreed, Cells: cells, Cell63: cell63, Texts: len(chains),
	}, nil
}
