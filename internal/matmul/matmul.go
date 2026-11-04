// Package matmul is the hodgelinear packing check. Tile is a name-keyed fixture
// for that check. A Hodge-Kuramoto reply reads the checkpoint, not this tile.
//
// The GPU program is the kernels as written: dot4_soft.wgsl, then
// hodgelinear.wgsl. Dawn loads that sequence the way its multi-file loader
// does, and dispatches fn main. Weights are packed the way hodgelinear.wgsl
// describes: four output rows share a byte, four inputs share a u32, a code
// is weight+1, and the stored result is (code - CENTER) times the activation.
package matmul

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"modelrun/internal/dawn"
	"modelrun/internal/paths"
)

// Default tile for a chat step. K and N are both multiples of 4. N is the
// output rows; the kernel's NP is N/4 packed rows.
const (
	DefaultK = 64
	DefaultN = 32
)

// Acts repeats the prompt bytes as int8 activations of length K.
func Acts(prompt string, K int) []int8 {
	b := []byte(prompt)
	if len(b) == 0 {
		b = []byte{0}
	}
	x := make([]int8, K)
	for i := range x {
		x[i] = int8(b[i%len(b)])
	}
	return x
}

// Tile is a stable ternary matrix for a model name, row-major [N][K],
// each value in {-1, 0, +1}. It is not the checkpoint. The BNZ weights stay
// in the EncryptidAI model directory.
func Tile(name string, N, K int) []int8 {
	var s uint32 = 2166136261
	for i := 0; i < len(name); i++ {
		s ^= uint32(name[i])
		s *= 16777619
	}
	w := make([]int8, N*K)
	for i := range w {
		s = s*1664525 + 1013904223
		w[i] = int8(int(s%3) - 1)
	}
	return w
}

// CPU is the same contraction as hodgelinear.wgsl with CENTER 1:
// y[row] = xs * ws * sum_k x[k] * W[row][k].
func CPU(x, w []int8, N, K int, xs, ws float32) []float32 {
	y := make([]float32, N)
	for row := 0; row < N; row++ {
		var acc int
		base := row * K
		for k := 0; k < K; k++ {
			acc += int(x[k]) * int(w[base+k])
		}
		y[row] = float32(acc) * xs * ws
	}
	return y
}

// packActs stores four int8 activations in a u32, first input in the low byte.
// dot4 reads byte j with shift 24-8j, which is the low byte at j=0.
// PackActs is packActs for callers outside this package.
func PackActs(x []int8) []uint32 { return packActs(x) }

func packActs(x []int8) []uint32 {
	kw := len(x) / 4
	out := make([]uint32, kw)
	for g := 0; g < kw; g++ {
		var word uint32
		for j := 0; j < 4; j++ {
			word |= uint32(byte(x[g*4+j])) << (8 * j)
		}
		out[g] = word
	}
	return out
}

// packWeights packs W[N][K] into NP*(K/4) words.
// Packed row r holds output rows r, r+NP, r+2NP, r+3NP.
// Input k's byte has those four codes in bit pairs 0, 2, 4, 6.
// A word holds four consecutive inputs, low byte first.
func packWeights(w []int8, N, K int) []uint32 {
	np := N / 4
	kw := K / 4
	out := make([]uint32, np*kw)
	for r := 0; r < np; r++ {
		for g := 0; g < kw; g++ {
			var word uint32
			for j := 0; j < 4; j++ {
				k := g*4 + j
				var b byte
				for i := 0; i < 4; i++ {
					row := r + i*np
					code := byte(int(w[row*K+k]) + 1)
					b |= (code & 3) << (2 * i)
				}
				word |= uint32(b) << (8 * j)
			}
			out[r*kw+g] = word
		}
	}
	return out
}

func writeU32(path string, w []uint32) error {
	b := make([]byte, len(w)*4)
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return os.WriteFile(path, b, 0o644)
}

func writeF32(path string, w []float32) error {
	b := make([]byte, len(w)*4)
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return os.WriteFile(path, b, 0o644)
}

func readF32(path string, n int) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < n*4 {
		return nil, fmt.Errorf("%s: %d bytes, want %d", path, len(b), n*4)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out, nil
}

// program is dot4_soft.wgsl followed by hodgelinear.wgsl, unchanged.
func program(dir string) (string, error) {
	k := paths.Default().KernelsDir()
	dot, err := os.ReadFile(filepath.Join(k, "dot4_soft.wgsl"))
	if err != nil {
		return "", err
	}
	lin, err := os.ReadFile(filepath.Join(k, "hodgelinear.wgsl"))
	if err != nil {
		return "", err
	}
	p := filepath.Join(dir, "hodgelinear.wgsl")
	body := append(append(dot, '\n'), lin...)
	if err := os.WriteFile(p, body, 0o644); err != nil {
		return "", err
	}
	return p, nil
}

// GPU dispatches hodgelinear on Dawn. N and K must be multiples of 4, and
// len(x) must be K, len(w) must be N*K. xs and ws are the activation and
// weight scales. The returned string is Dawn's stdout.
func GPU(x, w []int8, N, K int, xs, ws float32) ([]float32, string, error) {
	if K < 4 || N < 4 || K%4 != 0 || N%4 != 0 {
		return nil, "", fmt.Errorf("matmul: N=%d K=%d, both must be multiples of 4", N, K)
	}
	if len(x) != K || len(w) != N*K {
		return nil, "", fmt.Errorf("matmul: x %d w %d, want K=%d N*K=%d", len(x), len(w), K, N*K)
	}
	dir, err := os.MkdirTemp("", "modelrun-")
	if err != nil {
		return nil, "", err
	}
	defer os.RemoveAll(dir)
	prog, err := program(dir)
	if err != nil {
		return nil, "", err
	}
	np := N / 4
	if err := writeU32(filepath.Join(dir, "xq.bin"), packActs(x)); err != nil {
		return nil, "", err
	}
	if err := writeU32(filepath.Join(dir, "w.bin"), packWeights(w, N, K)); err != nil {
		return nil, "", err
	}
	if err := writeF32(filepath.Join(dir, "xs.bin"), []float32{xs}); err != nil {
		return nil, "", err
	}
	if err := writeF32(filepath.Join(dir, "ws.bin"), []float32{ws}); err != nil {
		return nil, "", err
	}
	if err := writeF32(filepath.Join(dir, "o.bin"), make([]float32, N)); err != nil {
		return nil, "", err
	}
	groups := (np + 7) / 8
	out := filepath.Join(dir, "out.bin")
	script := strings.Join([]string{
		"prog " + prog,
		"load xq " + filepath.Join(dir, "xq.bin"),
		"load w " + filepath.Join(dir, "w.bin"),
		"load xs " + filepath.Join(dir, "xs.bin"),
		"load ws " + filepath.Join(dir, "ws.bin"),
		"load o " + filepath.Join(dir, "o.bin"),
		"run main -d " + strconv.Itoa(groups) + " K=" + strconv.Itoa(K) + " NP=" + strconv.Itoa(np) + " CENTER=1",
		"save o " + out,
	}, "; ")
	res, err := dawn.Run(script)
	if err != nil {
		return nil, res.Stdout, err
	}
	y, err := readF32(out, N)
	if err != nil {
		return nil, res.Stdout, err
	}
	return y, res.Stdout, nil
}
