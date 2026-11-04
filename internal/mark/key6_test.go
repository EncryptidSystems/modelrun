package mark

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Published key from the first three poems. 2 bits per cell, cell 0 in the
// low bits: 0 disagree, 1 shared plus, 2 shared minus.
const publishedKey = "51096889980560592506049222168258"

func TestKeyHoldsSix(t *testing.T) {
	model := `C:\Users\grego\models\dev\hodge-3.5-instruct`
	dir := `C:\Users\grego\watermarks`
	if _, err := os.Stat(filepath.Join(model, "model.cochain.safetensors")); err != nil {
		t.Skip(err)
	}
	c, err := Open(filepath.Join(model, "tokenizer.json"), filepath.Join(model, "tokenizer_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"poem-1.txt", "poem-2.txt", "poem.txt", "poem-3.txt", "poem-4.txt", "poem-5.txt"}
	ch := make([]Chain, len(names))
	seen := map[string]string{}
	for i, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		ch[i] = c.Chain(name, string(b))
		t.Logf("%s  words %d  alignment %.3f  signs %s", ch[i].Name, len(ch[i].Words), ch[i].Alignment, hex.EncodeToString(packSigns(ch[i].Centroid)))
		for _, w := range ch[i].Words {
			k := strings.ToLower(w)
			if prev, ok := seen[k]; ok {
				t.Logf("repeat %q in %s and %s", w, prev, name)
			} else {
				seen[k] = name
			}
		}
	}
	code := decodeKey(publishedKey)
	keyed := 0
	for _, v := range code {
		if v != 0 {
			keyed++
		}
	}
	t.Logf("published key cells %d", keyed)
	for i := range ch {
		hit, miss := 0, []int{}
		for g, v := range code {
			if v == 0 {
				continue
			}
			got := 1
			if ch[i].Centroid[g] < 0 {
				got = 2
			}
			if got == v {
				hit++
			} else {
				miss = append(miss, g)
			}
		}
		t.Logf("read %s  %d/%d  miss %v", ch[i].Name, hit, keyed, miss)
	}
	// Poems 1–3 read every published cell. Poems 3–5 of the file set do not.
	// This is a Hodge-cochain reading. It is not a SynthID inversion.
	first3 := hex.EncodeToString(packMask(ch[:3]))
	all6 := hex.EncodeToString(packMask(ch))
	t.Logf("first3 %s", first3)
	t.Logf("all6   %s", all6)
	if first3 != publishedKey {
		t.Errorf("first three no longer pack to the published key")
	}
	grew := []int{}
	for g := range code {
		if code[g] != 0 {
			continue
		}
		s := ch[0].Centroid[g] >= 0
		ok := true
		for i := 1; i < len(ch); i++ {
			if (ch[i].Centroid[g] >= 0) != s {
				ok = false
				break
			}
		}
		if ok {
			grew = append(grew, g)
		}
	}
	t.Logf("cells the first three left open that all six share: %v", grew)
	if all6 == publishedKey {
		t.Errorf("six-poem packing matched the three-poem key; the measured result was a shorter agreement")
	}
	var key uint64
	cell63 := 0
	nAgree := 0
	for g := 0; g < len(ch[0].Centroid); g++ {
		s := signBit(ch[0].Centroid[g])
		ok := true
		for i := 1; i < len(ch); i++ {
			if signBit(ch[i].Centroid[g]) != s {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		nAgree++
		if g == 63 {
			cell63 = s
			continue
		}
		if s > 0 {
			key |= 1 << uint(g)
		}
	}
	got, err := KeyFrom(ch)
	if err != nil {
		t.Fatal(err)
	}
	if got.Key != 4638780189597239321 || got.Code != "41096880000120190102049022168258" {
		t.Fatalf("corpus key %d %s", got.Key, got.Code)
	}
	t.Logf("cochain key %d  agreed cells %d  cell63 %d", key, nAgree, cell63)
	if key >= 1<<63 {
		t.Fatalf("key %d is outside [0, 2^63)", key)
	}
}

func decodeKey(s string) []int {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	o := make([]int, 64)
	for g := range o {
		o[g] = int((b[g/4] >> uint((g%4)*2)) & 3)
	}
	return o
}
