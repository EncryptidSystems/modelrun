// Command cochainkey extracts the shared sign key of a corpus.
// It is not modelrun. It does not use the catalog.
//
//	cochainkey --config FILE CORPUS
//
// FILE is a models list. model is any safetensors file. tokenizer and
// tokenizer_config are the paths in that file. CORPUS is a folder of .txt texts.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"modelrun/internal/mark"
	"modelrun/internal/tokenizers"
)

func main() {
	config := flag.String("config", "", "yaml config")
	tensor := flag.String("tensor", "", "tensor name when the safetensors file has more than one")
	flag.Parse()
	if *config == "" || flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "cochainkey --config FILE CORPUS")
		os.Exit(2)
	}
	if err := run(*config, flag.Arg(0), *tensor); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(config, dir, tensor string) error {
	b, err := os.ReadFile(config)
	if err != nil {
		return err
	}
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	var doc struct {
		Models []struct {
			Model           string `yaml:"model"`
			Tokenizer       string `yaml:"tokenizer"`
			TokenizerConfig string `yaml:"tokenizer_config"`
			Config          string `yaml:"config"`
			SpecialTokens   string `yaml:"special_tokens_map"`
			Generation      string `yaml:"generation_config"`
			Tensor          string `yaml:"tensor"`
		} `yaml:"models"`
	}
	if err := dec.Decode(&doc); err != nil {
		return fmt.Errorf("%s: %v", config, err)
	}
	if len(doc.Models) == 0 {
		return fmt.Errorf("%s: models is empty", config)
	}
	e := doc.Models[0]
	base := filepath.Dir(config)
	abs := func(p string) string {
		if p == "" || filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(base, p)
	}
	if tensor == "" {
		tensor = e.Tensor
	}
	data, rows, cols, name, err := mark.LoadMatrix(abs(e.Model), tensor)
	if err != nil {
		return err
	}
	tok, err := tokenizers.Load(abs(e.Tokenizer))
	if err != nil {
		return err
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var names []string
	for _, ent := range ents {
		if ent.IsDir() || !strings.EqualFold(filepath.Ext(ent.Name()), ".txt") {
			continue
		}
		names = append(names, ent.Name())
	}
	sort.Strings(names)
	if len(names) < 2 {
		return fmt.Errorf("%s needs at least two .txt texts", dir)
	}
	chains := make([]mark.Chain, 0, len(names))
	for _, textName := range names {
		body, err := os.ReadFile(filepath.Join(dir, textName))
		if err != nil {
			return err
		}
		var ids []int
		for _, id := range tok.Encode(string(body), true) {
			if tok.IsSpecial(id) {
				continue
			}
			ids = append(ids, id)
		}
		chains = append(chains, mark.Chain{
			Name:     textName,
			Centroid: mark.RowsCentroid(data, rows, cols, ids),
		})
	}
	k, err := mark.KeyFrom(chains)
	if err != nil {
		return err
	}
	fmt.Printf("tensor %s  %d x %d\n", name, rows, cols)
	fmt.Println(k.Key)
	fmt.Printf("code %s\n", k.Code)
	fmt.Printf("mask %s\n", k.Mask)
	fmt.Printf("agreed %d of %d\n", k.Agreed, k.Cells)
	switch k.Cell63 {
	case 1:
		fmt.Println("cell 63 plus")
	case -1:
		fmt.Println("cell 63 minus")
	default:
		fmt.Println("cell 63 open")
	}
	for _, ch := range chains {
		fmt.Println(ch.Name)
	}
	return nil
}
