package mark

import (
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSharedBytes(t *testing.T) {
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
	signBytes := make([][]byte, len(ch))
	for i := range ch {
		signBytes[i] = packSigns(ch[i].Centroid)
		t.Logf("signs %s  %s", ch[i].Name, hex.EncodeToString(signBytes[i]))
	}
	same := make([]byte, len(signBytes[0]))
	for i := range same {
		if signBytes[0][i] == signBytes[1][i] && signBytes[1][i] == signBytes[2][i] {
			same[i] = signBytes[0][i]
		}
	}
	t.Logf("sign bytes equal in full: %v", hex.EncodeToString(signBytes[0]) == hex.EncodeToString(signBytes[1]) && hex.EncodeToString(signBytes[1]) == hex.EncodeToString(signBytes[2]))
	var mask []byte
	for g := 0; g < C; g++ {
		s := signBit(ch[0].Centroid[g])
		ok := true
		for i := 1; i < len(ch); i++ {
			if signBit(ch[i].Centroid[g]) != s {
				ok = false
				break
			}
		}
		if ok {
			mask = append(mask, byte(s))
		} else {
			mask = append(mask, 0)
		}
	}
	t.Logf("shared sign mask packed %s", hex.EncodeToString(packMask(ch)))

	// Coarsest step that still puts every poem in the same int8 bin on a cell.
	// Report the packed int8 of the shared bins, 0 where the poems do not meet.
	for _, step := range []float64{1, 16, 32, 64, 128, 256, 512} {
		n := 0
		out := make([]byte, C)
		for g := 0; g < C; g++ {
			q := quant(ch[0].Centroid[g], step)
			ok := true
			for i := 1; i < len(ch); i++ {
				if quant(ch[i].Centroid[g], step) != q {
					ok = false
					break
				}
			}
			if ok {
				n++
				out[g] = byte(int8(q))
			}
		}
		t.Logf("step %4.0f  same bins %2d/64  bytes %s", step, n, hex.EncodeToString(out))
	}
}

func packSigns(v []float64) []byte {
	o := make([]byte, (len(v)+7)/8)
	for i, x := range v {
		if x >= 0 {
			o[i/8] |= 1 << uint(i%8)
		}
	}
	return o
}

func packMask(ch []Chain) []byte {
	o := make([]byte, (len(ch[0].Centroid)+7)/8)
	for g := range ch[0].Centroid {
		s := ch[0].Centroid[g] >= 0
		ok := true
		for i := 1; i < len(ch); i++ {
			if (ch[i].Centroid[g] >= 0) != s {
				ok = false
				break
			}
		}
		if !ok || !s {
			continue
		}
		o[g/8] |= 1 << uint(g%8)
	}
	// Negative shared cells are recorded in a second pass by leaving the bit 0
	// only when the shared sign is negative OR the cell disagrees. Disagreement
	// and a shared minus are then the same bit. Keep a parallel nibble? The
	// caller prints this as the positive-agreement mask. Build a 2-bit code instead.
	code := make([]byte, len(ch[0].Centroid))
	for g := range code {
		s0 := signBit(ch[0].Centroid[g])
		agree := true
		for i := 1; i < len(ch); i++ {
			if signBit(ch[i].Centroid[g]) != s0 {
				agree = false
			}
		}
		if !agree {
			code[g] = 0
			continue
		}
		if s0 > 0 {
			code[g] = 1
		} else {
			code[g] = 2
		}
	}
	// Pack 2 bits per cell, little-endian within each byte: cell 0 in the low bits.
	out := make([]byte, (len(code)*2+7)/8)
	for g, v := range code {
		shift := uint((g % 4) * 2)
		out[g/4] |= v << shift
	}
	_ = binary.LittleEndian
	return out
}

func signBit(v float64) int {
	if v >= 0 {
		return 1
	}
	return -1
}

func quant(v, step float64) int {
	return int(math.Round(v / step))
}
