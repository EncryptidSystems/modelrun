// Package paths locates Dawn, the WASMApollo WGSL kernels, and the EncryptidAI
// model tree. EncryptidAI is the stage-ai product under EncryptidSystems.
package paths

import (
	"os"
	"path/filepath"
)

// Roots is where ModelRun reads. Env vars override the defaults next to the
// user's home directory.
type Roots struct {
	Dawn     string // dawn.exe
	Apollo   string // WASMApollo, WGSL kernels and the Dawn driver
	Encrypt  string // EncryptidAI stage-ai tree (models, kernels)
}

// Default resolves the three trees.
func Default() Roots {
	home, _ := os.UserHomeDir()
	r := Roots{
		Dawn:    filepath.Join(home, "WASMApollo", "dawn", "dawn.exe"),
		Apollo:  filepath.Join(home, "WASMApollo"),
		Encrypt: filepath.Join(home, "EncryptidSystems", "stage-ai"),
	}
	if v := os.Getenv("MODELRUN_DAWN"); v != "" {
		r.Dawn = v
	}
	if v := os.Getenv("MODELRUN_APOLLO"); v != "" {
		r.Apollo = v
		if os.Getenv("MODELRUN_DAWN") == "" {
			r.Dawn = filepath.Join(v, "dawn", "dawn.exe")
		}
	}
	if v := os.Getenv("MODELRUN_ENCRYPTID"); v != "" {
		r.Encrypt = v
	}
	return r
}

// KernelsDir is EncryptidAI's WGSL kernel directory.
func (r Roots) KernelsDir() string { return filepath.Join(r.Encrypt, "kernels") }

// ModelsDir is EncryptidAI's model directory.
func (r Roots) ModelsDir() string { return filepath.Join(r.Encrypt, "models") }
