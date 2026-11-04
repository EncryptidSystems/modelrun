// Package session keeps one dawn host process open. The Hodge-Kuramoto
// dispatches share its buffers and its compiled pipelines.
package dawn

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
)

// HostExe is dawnhost.exe, the resident caller of Dawn's library API.
func HostExe() string {
	if v := os.Getenv("MODELRUN_DAWNHOST"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "ModelRun", "dawnhost", "dawnhost.exe")
}

// Session is one open Dawn device.
type Session struct {
	mu  sync.Mutex
	cmd *exec.Cmd
	in  io.WriteCloser
	out io.ReadCloser
	oN  int
}

// Open starts the host. The Dawn DLLs sit next to that exe.
func Open() (*Session, error) {
	exe := HostExe()
	if _, err := os.Stat(exe); err != nil {
		return nil, fmt.Errorf("dawn host %s: %w", exe, err)
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
	return &Session{cmd: cmd, in: in, out: out}, nil
}

// Close tells the host to exit.
func (s *Session) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.in != nil {
		var n uint32 = 4
		_ = binary.Write(s.in, binary.LittleEndian, n)
		_, _ = s.in.Write([]byte("quit"))
		_ = s.in.Close()
		s.in = nil
	}
	if s.cmd != nil {
		_ = s.cmd.Wait()
		s.cmd = nil
	}
}

func (s *Session) round(cmd string, extra []byte) (uint32, []byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.in == nil {
		return 0, nil, fmt.Errorf("dawn host is closed")
	}
	if err := binary.Write(s.in, binary.LittleEndian, uint32(len(cmd))); err != nil {
		return 0, nil, err
	}
	if _, err := s.in.Write([]byte(cmd)); err != nil {
		return 0, nil, err
	}
	if extra != nil {
		if err := binary.Write(s.in, binary.LittleEndian, uint32(len(extra))); err != nil {
			return 0, nil, err
		}
		if _, err := s.in.Write(extra); err != nil {
			return 0, nil, err
		}
	}
	var rc, n uint32
	if err := binary.Read(s.out, binary.LittleEndian, &rc); err != nil {
		return 0, nil, fmt.Errorf("dawn host: %w", err)
	}
	if err := binary.Read(s.out, binary.LittleEndian, &n); err != nil {
		return 0, nil, fmt.Errorf("dawn host: %w", err)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(s.out, buf); err != nil {
		return 0, nil, fmt.Errorf("dawn host: %w", err)
	}
	if rc != 0 {
		msg := string(buf)
		if msg == "" {
			msg = "dawn host failed"
		}
		return rc, buf, fmt.Errorf("%s", msg)
	}
	return rc, buf, nil
}

// Exec runs one dawn command line (prog, info, clock, ...).
func (s *Session) Exec(line string) (string, error) {
	_, b, err := s.round(line, nil)
	return string(b), err
}

// Put copies bytes into a named buffer. len(b) must be a multiple of 4, or 0.
func (s *Session) Put(name string, b []byte) error {
	if len(b)%4 != 0 {
		return fmt.Errorf("put %s: %d bytes is not a multiple of 4", name, len(b))
	}
	_, _, err := s.round("put "+name, b)
	return err
}

// Get reads the first words of a buffer, or all of them when words is 0.
func (s *Session) Get(name string, words int) ([]byte, error) {
	_, b, err := s.round("get "+name+" "+strconv.Itoa(words), nil)
	return b, err
}

// Seq dispatches entry over groups workgroups. Pipelines are cached by the
// driver for the same overrides.
func (s *Session) Seq(entry string, groups int, ov string) error {
	_, _, err := s.round("seq "+entry+" "+strconv.Itoa(groups)+" "+ov, nil)
	return err
}

// EnsureOut makes the output buffer at least n floats. A later limb adds
// into that buffer, so it must already be large enough before the first limb.
func (s *Session) EnsureOut(n int) error {
	if n <= s.oN {
		return nil
	}
	if err := s.Put("o", make([]byte, n*4)); err != nil {
		return err
	}
	s.oN = n
	return nil
}

// Hodge dispatches hodgelinear.wgsl. w is the packed U8 matrix the shader
// reads. x is K int8 activations. xs and ws are the activation and limb scales.
// accum adds into the output buffer (a further digit limb).
func (s *Session) Hodge(w []byte, xs, ws float32, x []int8, n, k, center int, accum bool) ([]float32, error) {
	if n < 4 || k < 4 || n%4 != 0 || k%4 != 0 || len(x) != k {
		return nil, fmt.Errorf("hodge: N=%d K=%d x=%d", n, k, len(x))
	}
	if len(w) != (n/4)*k {
		return nil, fmt.Errorf("hodge: weight %d bytes, want %d", len(w), (n/4)*k)
	}
	if err := s.Put("w", w); err != nil {
		return nil, err
	}
	xq := make([]byte, k)
	for i, v := range x {
		xq[i] = byte(v)
	}
	if err := s.Put("xq", xq); err != nil {
		return nil, err
	}
	var xsb, wsb [4]byte
	binary.LittleEndian.PutUint32(xsb[:], math.Float32bits(xs))
	binary.LittleEndian.PutUint32(wsb[:], math.Float32bits(ws))
	if err := s.Put("xs", xsb[:]); err != nil {
		return nil, err
	}
	if err := s.Put("ws", wsb[:]); err != nil {
		return nil, err
	}
	if n > s.oN {
		if err := s.Put("o", make([]byte, n*4)); err != nil {
			return nil, err
		}
		s.oN = n
	}
	np := n / 4
	groups := (np + 7) / 8
	acc := "false"
	if accum {
		acc = "true"
	}
	ov := fmt.Sprintf("K=%d NP=%d CENTER=%d ACCUM=%s", k, np, center, acc)
	if err := s.Seq("main", groups, ov); err != nil {
		return nil, err
	}
	raw, err := s.Get("o", n)
	if err != nil {
		return nil, err
	}
	if len(raw) < n*4 {
		return nil, fmt.Errorf("hodge: output %d bytes, want %d", len(raw), n*4)
	}
	y := make([]float32, n)
	for i := range y {
		y[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return y, nil
}
