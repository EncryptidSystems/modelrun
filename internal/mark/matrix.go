package mark

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// LoadMatrix reads one 2D tensor from a safetensors file. The tensor name is
// used when it is set. Otherwise the file must contain one 2D tensor. Dtype
// may be I32, I16, I8, U8, F32, F16, or BF16.
func LoadMatrix(path, tensor string) (data []float64, rows, cols int, name string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, "", err
	}
	defer f.Close()
	var nb [8]byte
	if _, err := io.ReadFull(f, nb[:]); err != nil {
		return nil, 0, 0, "", fmt.Errorf("%s: %w", path, err)
	}
	n := binary.LittleEndian.Uint64(nb[:])
	if n == 0 || n > 1<<26 {
		return nil, 0, 0, "", fmt.Errorf("%s: header %d", path, n)
	}
	hb := make([]byte, n)
	if _, err := io.ReadFull(f, hb); err != nil {
		return nil, 0, 0, "", fmt.Errorf("%s: %w", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hb, &raw); err != nil {
		return nil, 0, 0, "", fmt.Errorf("%s: %w", path, err)
	}
	type info struct {
		Dtype string  `json:"dtype"`
		Shape []int   `json:"shape"`
		Off   []int64 `json:"data_offsets"`
	}
	var chosen info
	var names []string
	for k, body := range raw {
		if k == "__metadata__" {
			continue
		}
		var inf info
		if err := json.Unmarshal(body, &inf); err != nil || len(inf.Shape) != 2 {
			continue
		}
		names = append(names, k)
		if tensor != "" && k == tensor {
			chosen = inf
			name = k
		}
		if tensor == "" && name == "" {
			chosen = inf
			name = k
		}
	}
	if tensor != "" && name == "" {
		return nil, 0, 0, "", fmt.Errorf("%s has no tensor %s", path, tensor)
	}
	if tensor == "" && len(names) != 1 {
		return nil, 0, 0, "", fmt.Errorf("%s has %d two-dimensional tensors %v; set tensor", path, len(names), names)
	}
	rows, cols = chosen.Shape[0], chosen.Shape[1]
	if rows < 1 || cols < 1 || len(chosen.Off) != 2 {
		return nil, 0, 0, "", fmt.Errorf("%s: %s shape %v", path, name, chosen.Shape)
	}
	width := map[string]int{"I32": 4, "I16": 2, "I8": 1, "U8": 1, "F32": 4, "F16": 2, "BF16": 2}[chosen.Dtype]
	if width == 0 {
		return nil, 0, 0, "", fmt.Errorf("%s: %s dtype %s", path, name, chosen.Dtype)
	}
	if chosen.Off[1]-chosen.Off[0] != int64(rows*cols*width) {
		return nil, 0, 0, "", fmt.Errorf("%s: %s size", path, name)
	}
	b := make([]byte, rows*cols*width)
	if _, err := f.ReadAt(b, 8+int64(n)+chosen.Off[0]); err != nil {
		return nil, 0, 0, "", fmt.Errorf("%s: %w", path, err)
	}
	data = make([]float64, rows*cols)
	for i := range data {
		p := b[i*width : (i+1)*width]
		switch chosen.Dtype {
		case "I32":
			data[i] = float64(int32(binary.LittleEndian.Uint32(p)))
		case "I16":
			data[i] = float64(int16(binary.LittleEndian.Uint16(p)))
		case "I8":
			data[i] = float64(int8(p[0]))
		case "U8":
			data[i] = float64(p[0])
		case "F32":
			data[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(p)))
		case "F16":
			data[i] = float64(f16to32(binary.LittleEndian.Uint16(p)))
		case "BF16":
			data[i] = float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(p)) << 16))
		}
	}
	return data, rows, cols, name, nil
}

func f16to32(u uint16) float32 {
	sign := uint32(u&0x8000) << 16
	exp := uint32(u>>10) & 0x1f
	frac := uint32(u & 0x3ff)
	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign
		} else {
			exp = 127 - 14
			for frac&0x400 == 0 {
				frac <<= 1
				exp--
			}
			bits = sign | (exp << 23) | ((frac & 0x3ff) << 13)
		}
	case 31:
		bits = sign | 0x7f800000 | (frac << 13)
	default:
		bits = sign | ((exp + 127 - 15) << 23) | (frac << 13)
	}
	return math.Float32frombits(bits)
}

// RowsCentroid is the mean of the token rows after the vocabulary mean is removed.
func RowsCentroid(data []float64, rows, cols int, ids []int) []float64 {
	mu := make([]float64, cols)
	for id := 0; id < rows; id++ {
		base := id * cols
		for g := 0; g < cols; g++ {
			mu[g] += data[base+g]
		}
	}
	for g := range mu {
		mu[g] /= float64(rows)
	}
	cent := make([]float64, cols)
	n := 0
	for _, id := range ids {
		if id < 0 || id >= rows {
			continue
		}
		base := id * cols
		for g := 0; g < cols; g++ {
			cent[g] += data[base+g] - mu[g]
		}
		n++
	}
	if n == 0 {
		return cent
	}
	for g := range cent {
		cent[g] /= float64(n)
	}
	return cent
}
