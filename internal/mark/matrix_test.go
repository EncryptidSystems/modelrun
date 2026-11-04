package mark

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadMatrixAnyName(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "weights.safetensors")
	hdr := map[string]any{
		"embed": map[string]any{
			"dtype":        "F32",
			"shape":        []int{2, 2},
			"data_offsets": []int{0, 16},
		},
	}
	hb, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(hb)))
	if _, err := f.Write(n[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(hb); err != nil {
		t.Fatal(err)
	}
	vals := []float32{1, -2, 3, -4}
	for _, v := range vals {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		if _, err := f.Write(b[:]); err != nil {
			t.Fatal(err)
		}
	}
	f.Close()
	data, rows, cols, name, err := LoadMatrix(path, "")
	if err != nil {
		t.Fatal(err)
	}
	if name != "embed" || rows != 2 || cols != 2 || data[0] != 1 || data[3] != -4 {
		t.Fatalf("got %s %d %d %v", name, rows, cols, data)
	}
}
