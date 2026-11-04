// Package catalog lists EncryptidAI models and the WASMApollo LUT kernels
// Dawn can load.
package catalog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"modelrun/internal/paths"

	"gopkg.in/yaml.v3"
)

// Family is how a run is dispatched.
const (
	FamilyHodge = "hodge" // Hodge-Kuramoto runtime, hodgelinear on Dawn
	FamilyLUT   = "lut"   // dawn prog of a WGSL file
	FamilyClock = "clock" // dawn clock, the fabric LUT self-check
)

// Model is one row in `modelrun list`.
type Model struct {
	Name             string `json:"name"`
	Family           string `json:"family"`
	Path             string `json:"path,omitempty"`
	Config           string `json:"config,omitempty"`
	Tokenizer        string `json:"tokenizer,omitempty"`
	TokenizerConfig  string `json:"tokenizer_config,omitempty"`
	GenerationConfig string `json:"generation_config,omitempty"`
	Detail           string `json:"detail,omitempty"`
	Layers           int    `json:"layers,omitempty"`
	Hidden           int    `json:"hidden,omitempty"`
	Vocab            int    `json:"vocab,omitempty"`
	Kind             string `json:"kind,omitempty"`
	Bytes            int64  `json:"bytes,omitempty"`
}

// List returns the Dawn clock, then LUT kernels. A checkpoint is not discovered.
// modelrun run --config names its files.
func List() ([]Model, error) {
	r := paths.Default()
	var out []Model
	out = append(out, Model{
		Name:   "dawn/clock",
		Family: FamilyClock,
		Path:   r.Dawn,
		Detail: "fabric_fused, fabric_batch, fabric_sat",
		Kind:   "lut",
	})
	kernels, err := luts(r)
	if err != nil {
		return nil, err
	}
	out = append(out, kernels...)
	return out, nil
}

// Get finds a model by name.
func Get(name string) (Model, bool, error) {
	all, err := List()
	if err != nil {
		return Model{}, false, err
	}
	for _, m := range all {
		if m.Name == name {
			return m, true, nil
		}
	}
	return Model{}, false, nil
}

// Load reads one config file. Every path is a field in that file.
func Load(path string) (Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Model{}, err
	}
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	var doc struct {
		Models []struct {
			Model            string `yaml:"model"`
			Tokenizer        string `yaml:"tokenizer"`
			TokenizerConfig  string `yaml:"tokenizer_config"`
			Config           string `yaml:"config"`
			SpecialTokensMap string `yaml:"special_tokens_map"`
			GenerationConfig string `yaml:"generation_config"`
		} `yaml:"models"`
	}
	if err := dec.Decode(&doc); err != nil {
		return Model{}, fmt.Errorf("%s: %v", path, err)
	}
	if len(doc.Models) == 0 {
		return Model{}, fmt.Errorf("%s: models is empty", path)
	}
	e := doc.Models[0]
	dir := filepath.Dir(path)
	abs := func(p string) string {
		if p == "" || filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(dir, p)
	}
	weight := abs(e.Model)
	cfgPath := abs(e.Config)
	tok := abs(e.Tokenizer)
	tokCfg := abs(e.TokenizerConfig)
	gen := abs(e.GenerationConfig)
	for _, pair := range []struct{ what, p string }{
		{"model", weight},
		{"config", cfgPath},
		{"tokenizer", tok},
		{"tokenizer_config", tokCfg},
	} {
		if pair.p == "" {
			return Model{}, fmt.Errorf("%s: %s is not set", path, pair.what)
		}
		if _, err := os.Stat(pair.p); err != nil {
			return Model{}, fmt.Errorf("%s: %s: %v", path, pair.what, err)
		}
	}
	if gen != "" {
		if _, err := os.Stat(gen); err != nil {
			return Model{}, fmt.Errorf("%s: generation_config: %v", path, err)
		}
	}
	cfgb, err := os.ReadFile(cfgPath)
	if err != nil {
		return Model{}, err
	}
	var cfg struct {
		Hidden int `json:"hidden_size"`
		Layers int `json:"num_hidden_layers"`
		Vocab  int `json:"vocab_size"`
	}
	if err := json.Unmarshal(cfgb, &cfg); err != nil {
		return Model{}, fmt.Errorf("%s: %v", cfgPath, err)
	}
	st, err := os.Stat(weight)
	if err != nil {
		return Model{}, err
	}
	return Model{
		Name:             path,
		Family:           FamilyHodge,
		Path:             weight,
		Config:           cfgPath,
		Tokenizer:        tok,
		TokenizerConfig:  tokCfg,
		GenerationConfig: gen,
		Detail:           filepath.Base(path),
		Layers:           cfg.Layers,
		Hidden:           cfg.Hidden,
		Vocab:            cfg.Vocab,
		Kind:             "hodge",
		Bytes:            st.Size(),
	}, nil
}

func luts(r paths.Roots) ([]Model, error) {
	dirs := []string{
		r.Apollo,
		filepath.Join(r.Apollo, "declut", "wgsl"),
		r.KernelsDir(),
	}
	seen := map[string]bool{}
	var out []Model
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".wgsl") {
				continue
			}
			if !wanted(e.Name()) {
				continue
			}
			name := "apollo/" + strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
			if seen[name] {
				continue
			}
			seen[name] = true
			p := filepath.Join(d, e.Name())
			var bytes int64
			if st, err := os.Stat(p); err == nil {
				bytes = st.Size()
			}
			out = append(out, Model{
				Name:   name,
				Family: FamilyLUT,
				Path:   p,
				Detail: "WGSL",
				Kind:   "lut",
				Bytes:  bytes,
			})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

func wanted(name string) bool {
	n := strings.ToLower(name)
	for _, s := range []string{"lut", "fabric", "ternary", "hodge", "embed", "dot4", "kuramoto"} {
		if strings.Contains(n, s) {
			return true
		}
	}
	return false
}
