package synthid

import (
	"os"
	"path/filepath"
	"testing"

	"modelrun/internal/tokenizers"
)

func TestGemma4PoemIDs(t *testing.T) {
	tokPath := `C:\Users\grego\models\gemma\gemma4\tokenizer.json`
	dir := `C:\Users\grego\watermarks`
	if _, err := os.Stat(tokPath); err != nil {
		t.Skip(err)
	}
	tok, err := tokenizers.Load(tokPath)
	if err != nil {
		t.Fatal(err)
	}
	if tok.Size() != 262144 {
		t.Fatalf("vocab %d, Gemma 4 is 262144", tok.Size())
	}
	if got := tok.Encode(" the", false); len(got) != 1 || got[0] != 506 {
		t.Fatalf("▁the encoded %v", got)
	}
	if tok.Decode([]int{506}, false) != " the" {
		t.Fatalf("decode ▁the %q", tok.Decode([]int{506}, false))
	}
	names := []string{"poem-1.txt", "poem-2.txt", "poem.txt", "poem-3.txt", "poem-4.txt", "poem-5.txt"}
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		ids := tok.Encode(string(b), true)
		if back := tok.Decode(ids, true); back != string(b) {
			t.Fatalf("%s did not round-trip (%d ids)", name, len(ids))
		}
		head := ids
		if len(head) > 12 {
			head = head[:12]
		}
		t.Logf("%s  %d token ids  first %v", name, len(ids), head)
	}
}
