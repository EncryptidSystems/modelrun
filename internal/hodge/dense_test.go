package hodge

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"modelrun/internal/paths"
	"modelrun/internal/rt"
)

func f32bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

func f32s(b []byte) []float32 {
	o := make([]float32, len(b)/4)
	for i := range o {
		o[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return o
}

func TestDenseGPU(t *testing.T) {
	if _, err := os.Stat(rt.Exe()); err != nil {
		t.Skip(err)
	}
	kdir := paths.Default().KernelsDir()
	normSrc, err := os.ReadFile(filepath.Join(kdir, "norm.wgsl"))
	if err != nil {
		t.Fatal(err)
	}
	denseSrc, err := os.ReadFile(filepath.Join(kdir, "dense.wgsl"))
	if err != nil {
		t.Fatal(err)
	}
	dev, err := rt.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	if err := dev.Shader("norm.wgsl", string(normSrc)); err != nil {
		t.Fatal(err)
	}
	if err := dev.Shader("dense.wgsl", string(denseSrc)); err != nil {
		t.Fatal(err)
	}

	// x = [1, 0, 0, 0], norm weight 2 (not the all-ones sub-norm).
	// Ten rows so the second workgroup (rows 8 and 9) has to run.
	x := f32bytes([]float32{1, 0, 0, 0})
	nw := f32bytes([]float32{2, 2, 2, 2})
	rows := make([]float32, 10*4)
	rows[0], rows[1] = 1, 2
	rows[6] = 1
	rows[32] = 1
	rows[37] = 1
	w := bf16bytes(rows...)
	if _, err := dev.Buf("a", x, len(x)); err != nil {
		t.Fatal(err)
	}
	if _, err := dev.Buf("b", x, len(x)); err != nil {
		t.Fatal(err)
	}
	if _, err := dev.Buf("nw", nw, len(nw)); err != nil {
		t.Fatal(err)
	}
	if _, err := dev.Buf("xq", make([]byte, 4), 4); err != nil {
		t.Fatal(err)
	}
	if _, err := dev.Buf("xs", make([]byte, 16), 16); err != nil {
		t.Fatal(err)
	}
	yid, err := dev.Buf("y", make([]byte, 16), 16)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dev.Buf("w", w, len(w)); err != nil {
		t.Fatal(err)
	}
	if _, err := dev.Buf("zbias", []byte{0, 0, 0, 0}, 4); err != nil {
		t.Fatal(err)
	}
	oid, err := dev.Buf("o", make([]byte, 40), 40)
	if err != nil {
		t.Fatal(err)
	}
	nk := []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRW}
	np, err := dev.Pipe("normD", "norm.wgsl", "main", "K=4 EPS=1e-5 GEMMA=0 MODE=0 ACT=0", nk)
	if err != nil {
		t.Fatal(err)
	}
	ng, err := dev.Group("norm", "normD", []rt.Bind{
		{"a", 0, 0}, {"b", 0, 0}, {"nw", 0, 0}, {"xq", 0, 0}, {"xs", 0, 16}, {"y", 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}
	dp, err := dev.Pipe("dense", "dense.wgsl", "main", "N=10 K=4 BIAS=0", []byte{rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRO})
	if err != nil {
		t.Fatal(err)
	}
	dg, err := dev.Group("dense", "dense", []rt.Bind{
		{"y", 0, 0}, {"w", 0, 0}, {"o", 0, 0}, {"zbias", 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := dev.Run([]rt.Op{
		{Tag: rt.OpDispatch, Pipe: np, Group: ng, X: 1, Y: 1, Z: 1},
		{Tag: rt.OpDispatch, Pipe: dp, Group: dg, X: 2, Y: 1, Z: 1},
	}, []rt.Get{{Buf: yid, N: 16}, {Buf: oid, N: 40}})
	if err != nil {
		t.Fatal(err)
	}
	y := f32s(out[:16])
	o := f32s(out[16:])
	// mean(x^2)=0.25, inv=1/sqrt(0.25+1e-5), gain=2.
	wantY := float32(2 / math.Sqrt(0.25+1e-5))
	if math.Abs(float64(y[0]-wantY)) > 1e-3 || y[1] != 0 || y[2] != 0 || y[3] != 0 {
		t.Fatalf("y %v want %v", y, wantY)
	}
	// Row 0 is [1, 2, 0, 0]. Row 8 is in the second workgroup and is [1, 0, 0, 0].
	if math.Abs(float64(o[0]-y[0])) > 1e-2 || math.Abs(float64(o[1])) > 1e-4 || math.Abs(float64(o[8]-y[0])) > 1e-2 || math.Abs(float64(o[9])) > 1e-4 {
		t.Fatalf("o %v y %v", o, y)
	}
}
