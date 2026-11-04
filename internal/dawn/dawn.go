// Package dawn drives WASMApollo's Dawn LUT runner. One call is one dawn.exe
// invocation: semicolon-separated commands, the same script run_wgsl.py sends.
// The WGSL files are loaded as written.
package dawn

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"modelrun/internal/paths"
)

// Result is one dawn.exe run.
type Result struct {
	Stdout  string
	Elapsed time.Duration
}

// Run executes script on dawn.exe. script is the command text after the
// program name, for example "info" or "prog a.wgsl; run main -d 1".
func Run(script string) (Result, error) {
	exe := paths.Default().Dawn
	if _, err := os.Stat(exe); err != nil {
		return Result{}, fmt.Errorf("dawn runner %s: %w", exe, err)
	}
	start := time.Now()
	cmd := exec.Command(exe, script)
	out, err := cmd.CombinedOutput()
	res := Result{Stdout: string(out), Elapsed: time.Since(start)}
	if err != nil {
		msg := strings.TrimSpace(res.Stdout)
		if msg == "" {
			msg = err.Error()
		}
		return res, fmt.Errorf("dawn: %s", msg)
	}
	return res, nil
}

// Info is `dawn info`: the adapter the LUT runner opened.
func Info() (string, error) {
	res, err := Run("info")
	return strings.TrimSpace(res.Stdout), err
}
