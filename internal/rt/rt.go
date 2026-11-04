// Package rt talks to dawnrt. Buffers and pipelines stay on the device.
// A Run is one submit: the weights are not written again.
package rt

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

const (
	opShader = 1
	opBuf    = 2
	opPipe   = 3
	opGroup  = 4
	opRun    = 5
	opWrite  = 7

	KindRO  = 0
	KindRW  = 1
	KindUni = 2

	OpDispatch = 1
	OpCopy     = 2
)

// Exe is dawnrt.exe.
func Exe() string {
	if v := os.Getenv("MODELRUN_DAWNRT"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "ModelRun", "dawnrt", "dawnrt.exe")
}

// Dev is one device. Ids are the host's.
type Dev struct {
	mu  sync.Mutex
	cmd *exec.Cmd
	in  io.WriteCloser
	out io.ReadCloser
}

// Open starts the runtime.
func Open() (*Dev, error) {
	exe := Exe()
	if _, err := os.Stat(exe); err != nil {
		return nil, fmt.Errorf("dawnrt %s: %w", exe, err)
	}
	cmd := exec.Command(exe)
	cmd.Stderr = os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &Dev{cmd: cmd, in: in, out: out}, nil
}

// Close stops the process.
func (d *Dev) Close() {
	if d == nil || d.in == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	var n uint32 = 4
	_ = binary.Write(d.in, binary.LittleEndian, n)
	_, _ = d.in.Write([]byte("quit"))
	_ = d.in.Close()
	d.in = nil
	if d.cmd != nil {
		_ = d.cmd.Wait()
	}
}

func (d *Dev) call(parts ...[]byte) ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	var n uint32
	for _, p := range parts {
		n += uint32(len(p))
	}
	if err := binary.Write(d.in, binary.LittleEndian, n); err != nil {
		return nil, err
	}
	for _, p := range parts {
		if _, err := d.in.Write(p); err != nil {
			return nil, err
		}
	}
	var rc, m uint32
	if err := binary.Read(d.out, binary.LittleEndian, &rc); err != nil {
		return nil, err
	}
	if err := binary.Read(d.out, binary.LittleEndian, &m); err != nil {
		return nil, err
	}
	buf := make([]byte, m)
	if _, err := io.ReadFull(d.out, buf); err != nil {
		return nil, err
	}
	if rc != 0 {
		return nil, fmt.Errorf("%s", string(buf))
	}
	return buf, nil
}

func idOf(b []byte) (uint32, error) {
	if len(b) < 4 {
		return 0, fmt.Errorf("dawnrt: short id")
	}
	return binary.LittleEndian.Uint32(b), nil
}

func u32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

func u64(v uint64) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, v)
	return b
}

func str(s string) []byte {
	b := u32(uint32(len(s)))
	return append(b, s...)
}

// Shader compiles a WGSL module.
func (d *Dev) Shader(name, src string) error {
	_, err := d.call(u32(opShader), str(name), u32(uint32(len(src))), []byte(src))
	return err
}

// Buf creates a buffer and writes data at the start. size 0 means len(data).
func (d *Dev) Buf(name string, data []byte, size int) (uint32, error) {
	if size < len(data) {
		size = len(data)
	}
	b, err := d.call(u32(opBuf), str(name), u64(uint64(size)), u64(uint64(len(data))), data)
	if err != nil {
		return 0, err
	}
	return idOf(b)
}

// Write replaces bytes in an existing buffer.
func (d *Dev) Write(name string, off int, data []byte) error {
	_, err := d.call(u32(opWrite), str(name), u32(uint32(off)), u32(uint32(len(data))), data)
	return err
}

// Pipe compiles one entry. kinds are KindRO, KindRW, KindUni in binding order.
func (d *Dev) Pipe(name, shader, entry, overrides string, kinds []byte) (uint32, error) {
	parts := [][]byte{u32(opPipe), str(name), str(shader), str(entry), str(overrides), u32(uint32(len(kinds))), kinds}
	b, err := d.call(parts...)
	if err != nil {
		return 0, err
	}
	return idOf(b)
}

// Bind is one binding: buffer name, byte offset, byte size. Size 0 is the rest.
type Bind struct {
	Buf  string
	Off  int
	Size int
}

// Group binds buffers to a pipeline. The order is the binding order.
func (d *Dev) Group(name, pipe string, binds []Bind) (uint32, error) {
	parts := [][]byte{u32(opGroup), str(name), str(pipe), u32(uint32(len(binds)))}
	for _, b := range binds {
		parts = append(parts, str(b.Buf), u32(uint32(b.Off)), u32(uint32(b.Size)))
	}
	out, err := d.call(parts...)
	if err != nil {
		return 0, err
	}
	return idOf(out)
}

// Op is one recorded command. Tag OpDispatch uses Pipe, Group, X, Y, Z.
// Tag OpCopy uses Src, SrcOff, Dst, DstOff, N.
type Op struct {
	Tag            byte
	Pipe, Group    uint32
	X, Y, Z        uint32
	Src, Dst       uint32
	SrcOff, DstOff uint32
	N              uint32
}

// Get reads bytes after the submit.
type Get struct {
	Buf uint32
	Off int
	N   int
}

// Run submits ops as one command buffer, then reads gets.
func (d *Dev) Run(ops []Op, gets []Get) ([]byte, error) {
	body := make([]byte, 0, 8+len(ops)*24)
	body = append(body, u32(opRun)...)
	body = append(body, u32(uint32(len(ops)))...)
	for _, op := range ops {
		body = append(body, op.Tag)
		if op.Tag == OpCopy {
			body = append(body, u32(op.Src)...)
			body = append(body, u32(op.SrcOff)...)
			body = append(body, u32(op.Dst)...)
			body = append(body, u32(op.DstOff)...)
			body = append(body, u32(op.N)...)
		} else {
			body = append(body, u32(op.Pipe)...)
			body = append(body, u32(op.Group)...)
			body = append(body, u32(op.X)...)
			body = append(body, u32(op.Y)...)
			body = append(body, u32(op.Z)...)
		}
	}
	body = append(body, u32(uint32(len(gets)))...)
	for _, g := range gets {
		body = append(body, u32(g.Buf)...)
		body = append(body, u32(uint32(g.Off))...)
		body = append(body, u32(uint32(g.N))...)
	}
	return d.call(body)
}
