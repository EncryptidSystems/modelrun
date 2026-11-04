package hodge

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"hodgemix/bnz4"
)

type stTen struct {
	name, dtype string
	shape       []int
	b           []byte
}

func writeST(path string, tens []stTen) error {
	type info struct {
		Dtype string  `json:"dtype"`
		Shape []int   `json:"shape"`
		Off   []int64 `json:"data_offsets"`
	}
	h := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var data []byte
	var off int64
	for _, t := range tens {
		if p := int(off % 8); p != 0 {
			n := 8 - p
			data = append(data, make([]byte, n)...)
			off += int64(n)
		}
		h[t.name] = info{Dtype: t.dtype, Shape: t.shape, Off: []int64{off, off + int64(len(t.b))}}
		data = append(data, t.b...)
		off += int64(len(t.b))
	}
	hb, err := json.Marshal(h)
	if err != nil {
		return err
	}
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var nbuf [8]byte
	binary.LittleEndian.PutUint64(nbuf[:], uint64(len(hb)))
	if _, err := f.Write(nbuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(hb); err != nil {
		return err
	}
	_, err = f.Write(data)
	return err
}

func bf16bytes(vs ...float32) []byte {
	b := make([]byte, 2*len(vs))
	for i, v := range vs {
		binary.LittleEndian.PutUint16(b[2*i:], f32tobf16(v))
	}
	return b
}

func TestSafetensorsMatchesBNZ(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "model.safetensors")
	embed := bf16bytes(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	norm := bf16bytes(1, 2, 3, 4)
	w := []byte{0x1b, 0xe4, 0x00, 0x39}
	scale := bf16bytes(1)
	limb := []byte{0x00, 0x55, 0xaa, 0xff}
	limbScale := bf16bytes(0.5)
	radix := []byte{4, 0, 0, 0}
	const q = "model.layers.0.self_attn.q_proj"
	err := writeST(stPath, []stTen{
		{"model.embed_tokens.weight", "BF16", []int{4, 4}, embed},
		{"model.norm.weight", "BF16", []int{4}, norm},
		{q + ".weight", "U8", []int{1, 4}, w},
		{q + ".weight_scale", "BF16", []int{1}, scale},
		{q + ".weight_limb1", "U8", []int{1, 4}, limb},
		{q + ".weight_limb1_scale", "BF16", []int{1}, limbScale},
		{q + ".weight_limb1_radix", "I32", []int{1}, radix},
		{"bnzmix.bnz4." + q + ".weight", "U8", []int{8}, []byte{1, 2, 3, 4, 5, 6, 7, 8}},
	})
	if err != nil {
		t.Fatal(err)
	}
	bnzPath := filepath.Join(dir, "model.bnz")
	if _, err := bnz4.Compress(stPath, bnzPath, nil); err != nil {
		t.Fatal(err)
	}

	a, err := decodeAll(stPath)
	if err != nil {
		t.Fatal(err)
	}
	b, err := decodeAll(bnzPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := a["bnzmix.bnz4."+q+".weight"]; ok {
		t.Fatal("safetensors kept a bnzmix tensor")
	}
	if _, ok := b["bnzmix.bnz4."+q+".weight"]; ok {
		t.Fatal("bnz kept a bnzmix tensor")
	}
	if _, ok := a["__metadata__"]; ok {
		t.Fatal("metadata loaded as a tensor")
	}
	if layerCount(a) != 1 || layerCount(b) != 1 {
		t.Fatalf("layers %d %d", layerCount(a), layerCount(b))
	}
	ea, err := assembleEmbed(a, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	eb, err := assembleEmbed(b, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ea, embed) || !bytes.Equal(eb, embed) {
		t.Fatalf("embed safetensors %d bnz %d", len(ea), len(eb))
	}
	if !bytes.Equal(a["model.norm.weight"].b, b["model.norm.weight"].b) {
		t.Fatal("norm bytes differ")
	}
	la, err := buildLin(a, q, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	lb, err := buildLin(b, q, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(la.w, w) || !bytes.Equal(la.w, lb.w) || la.s != lb.s || la.s != 1 {
		t.Fatalf("weight scale %v %v", la.s, lb.s)
	}
	if len(la.limbs) != 1 || len(lb.limbs) != 1 || la.limbs[0].center != 1 || lb.limbs[0].center != 1 {
		t.Fatalf("limbs %+v %+v", la.limbs, lb.limbs)
	}
	if !bytes.Equal(la.limbs[0].w, limb) || !bytes.Equal(la.limbs[0].w, lb.limbs[0].w) || la.limbs[0].s != lb.limbs[0].s {
		t.Fatal("limb bytes differ")
	}
	if len(a[q+".weight"].shape) != 2 || a[q+".weight"].shape[0] != 1 || a[q+".weight"].shape[1] != 4 {
		t.Fatalf("shape %v", a[q+".weight"].shape)
	}
}

func TestFFNNormName(t *testing.T) {
	const p = "model.layers.0."
	if got := ffnNormName(map[string]got{
		p + "post_attention_layernorm.weight":  {},
		p + "pre_feedforward_layernorm.weight": {},
	}, p); got != p+"post_attention_layernorm.weight" {
		t.Fatal(got)
	}
	if got := ffnNormName(map[string]got{
		p + "pre_feedforward_layernorm.weight": {},
	}, p); got != p+"pre_feedforward_layernorm.weight" {
		t.Fatal(got)
	}
	if got := ffnNormName(map[string]got{}, p); got != p+"post_attention_layernorm.weight" {
		t.Fatal(got)
	}
}

func TestBuildLinFloat(t *testing.T) {
	const q = "model.layers.0.self_attn.q_proj"
	w := bf16bytes(
		1, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5,
		0, 0, 0, 1,
		1, 1, 1, 1,
	)
	bias := bf16bytes(1, 2, 3, 4)
	l, err := buildLin(map[string]got{
		q + ".weight": {b: w, shape: []int{4, 4}},
		q + ".bias":   {b: bias, shape: []int{4}},
	}, q, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(l.f, w) || l.w != nil || l.s != 0 || len(l.limbs) != 0 || !bytes.Equal(l.bias, bias) {
		t.Fatalf("float lin w %d f %d scale %v limbs %d", len(l.w), len(l.f), l.s, len(l.limbs))
	}
	x := []float32{1, 2, 3, 4}
	if d := bf16Row(l.f[:8], x); d != 1 {
		t.Fatal(d)
	}
	if d := bf16Row(l.f[8:16], x); d != 5 {
		t.Fatal(d)
	}
	row := []byte{0x80, 0x3f, 0x00, 0x40} // bf16 1 then bf16 2
	if d := bf16Row(row, []float32{10, 1}); d != 12 {
		t.Fatal(d)
	}

	dir := t.TempDir()
	p := filepath.Join(dir, "model.safetensors")
	f32 := make([]byte, 16)
	for i, v := range []float32{1, 2, 3, 4} {
		binary.LittleEndian.PutUint32(f32[i*4:], math.Float32bits(v))
	}
	if err := writeST(p, []stTen{{q + ".weight", "F32", []int{2, 2}, f32}}); err != nil {
		t.Fatal(err)
	}
	dec, err := decodeAll(p)
	if err != nil {
		t.Fatal(err)
	}
	lf, err := buildLin(dec, q, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(lf.f, bf16bytes(1, 2, 3, 4)) {
		t.Fatalf("%x", lf.f)
	}

	_, err = buildLin(map[string]got{
		q + ".weight": {b: make([]byte, 10), shape: []int{4, 4}},
	}, q, 4, 4)
	if err == nil {
		t.Fatal("expected a size error")
	}
	id, err := actID("silu")
	if err != nil || id != 1 {
		t.Fatal(id, err)
	}
	if id, err = actID("relu2"); err != nil || id != 0 {
		t.Fatal(id, err)
	}
	if id, err = actID("gelu_pytorch_tanh"); err != nil || id != 2 {
		t.Fatal(id, err)
	}
}

func TestSafetensorsFloatToBF16(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "model.safetensors")
	f32 := make([]byte, 16)
	binary.LittleEndian.PutUint32(f32[0:], math.Float32bits(1))
	binary.LittleEndian.PutUint32(f32[4:], math.Float32bits(-1))
	binary.LittleEndian.PutUint32(f32[8:], math.Float32bits(0.5))
	binary.LittleEndian.PutUint32(f32[12:], 0)
	round := make([]byte, 8)
	binary.LittleEndian.PutUint32(round[0:], 0x3f808000) // halfway, ties to even 0x3f80
	binary.LittleEndian.PutUint32(round[4:], 0x3f818000) // halfway, ties to even 0x3f82
	err := writeST(p, []stTen{
		{"model.norm.weight", "F32", []int{4}, f32},
		{"model.layers.0.input_layernorm.weight", "F32", []int{2}, round},
		{"model.layers.0.self_attn.q_proj.weight_scale", "F16", []int{1}, []byte{0x00, 0x3c}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeAll(p)
	if err != nil {
		t.Fatal(err)
	}
	w, err := vec(got, "model.norm.weight", 4, false)
	if err != nil {
		t.Fatal(err)
	}
	if w[0] != 1 || w[1] != -1 || w[2] != 0.5 || w[3] != 0 {
		t.Fatalf("%v", w)
	}
	rb := got["model.layers.0.input_layernorm.weight"].b
	if len(rb) != 4 || binary.LittleEndian.Uint16(rb[0:]) != 0x3f80 || binary.LittleEndian.Uint16(rb[2:]) != 0x3f82 {
		t.Fatalf("round %x", rb)
	}
	s, err := bf16one(got["model.layers.0.self_attn.q_proj.weight_scale"].b)
	if err != nil {
		t.Fatal(err)
	}
	if s != 1 {
		t.Fatal(s)
	}
}

func TestDecodeRejectsOtherFiles(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "nope.bin")
	if err := os.WriteFile(p, []byte("hello world, this is not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := decodeAll(p); err == nil {
		t.Fatal("expected an error")
	}
}
