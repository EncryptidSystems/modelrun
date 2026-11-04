package matmul

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"modelrun/internal/dawn"
	"modelrun/internal/paths"
)

func TestCPUMatchesHandDot(t *testing.T) {
	// One packed group: N=4, K=4. Row dots are small integers.
	x := []int8{1, 0, -1, 2}
	w := []int8{
		1, 0, -1, 1,
		0, 1, 0, -1,
		-1, -1, 1, 0,
		1, 1, 1, 1,
	}
	got := CPU(x, w, 4, 4, 1, 1)
	want := []float32{4, -2, -2, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("row %d got %v want %v", i, got[i], want[i])
		}
	}
}

func TestPackRoundTripShape(t *testing.T) {
	w := Tile("hodge-1", 8, 8)
	words := packWeights(w, 8, 8)
	if len(words) != 4 {
		t.Fatalf("words %d", len(words))
	}
	acts := packActs(Acts("tea", 8))
	if len(acts) != 2 {
		t.Fatalf("acts %d", len(acts))
	}
}

func TestGPUMatchesCPU(t *testing.T) {
	exe := paths.Default().Dawn
	if _, err := os.Stat(exe); err != nil {
		t.Skip(err)
	}
	const N, K = 8, 8
	x := Acts("boba", K)
	w := Tile("hodge-1", N, K)
	got, log, err := GPU(x, w, N, K, 1, 1)
	if err != nil {
		t.Fatalf("gpu: %v\n%s", err, log)
	}
	want := CPU(x, w, N, K, 1, 1)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("y[%d] gpu %v cpu %v\n%s", i, got[i], want[i], log)
		}
	}
}

func TestHostMatchesCPU(t *testing.T) {
	if _, err := os.Stat(dawn.HostExe()); err != nil {
		t.Skip(err)
	}
	const N, K = 8, 8
	x := Acts("boba", K)
	w := Tile("hodge-1", N, K)
	dir, err := os.MkdirTemp("", "modelrun-host-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	prog, err := program(dir)
	if err != nil {
		t.Fatal(err)
	}
	s, err := dawn.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.Exec("prog " + prog); err != nil {
		t.Fatal(err)
	}
	words := packWeights(w, N, K)
	wb := make([]byte, len(words)*4)
	for i, v := range words {
		binary.LittleEndian.PutUint32(wb[i*4:], v)
	}
	got, err := s.Hodge(wb, 1, 1, x, N, K, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	want := CPU(x, w, N, K, 1, 1)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("y[%d] host %v cpu %v", i, got[i], want[i])
		}
	}
}
