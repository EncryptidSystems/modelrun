package synthid

import (
	"os"
	"path/filepath"
	"testing"

	"modelrun/internal/tokenizers"
)

func TestVaultGemmaPoemIDs(t *testing.T) {
	tokPath := `C:\Users\grego\models\prod\vaultgemma\tokenizer.json`
	dir := `C:\Users\grego\watermarks`
	if _, err := os.Stat(tokPath); err != nil {
		t.Skip(err)
	}
	tok, err := tokenizers.Load(tokPath)
	if err != nil {
		t.Fatal(err)
	}
	if tok.Size() != 256000 {
		t.Fatalf("vocab %d, vaultgemma embedding is 256000 rows", tok.Size())
	}
	names := []string{"poem-1.txt", "poem-2.txt", "poem.txt", "poem-3.txt", "poem-4.txt", "poem-5.txt"}
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		ids := tok.Encode(string(b), true)
		back := tok.Decode(ids, true)
		if back != string(b) {
			t.Fatalf("%s did not round-trip (%d ids)", name, len(ids))
		}
		for _, id := range ids {
			if id < 0 || id >= tok.Size() {
				t.Fatalf("%s id %d outside the embedding", name, id)
			}
		}
		head := ids
		if len(head) > 12 {
			head = head[:12]
		}
		t.Logf("%s  %d token ids  first %v", name, len(ids), head)
	}
}
