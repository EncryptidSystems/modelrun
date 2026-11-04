// Command cochainimg draws the word cochains of the texts in a folder.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"modelrun/internal/mark"
)

func main() {
	model := flag.String("model", `C:\Users\grego\models\dev\hodge-3.5-instruct`, "model folder with the cochain and tokenizer")
	dir := flag.String("dir", `C:\Users\grego\watermarks`, "folder of .txt texts")
	out := flag.String("out", "", "png path (default: DIR/cochains.png)")
	flag.Parse()
	if *out == "" {
		*out = filepath.Join(*dir, "cochains.png")
	}
	if err := run(*model, *dir, *out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(*out)
}

func run(model, dir, out string) error {
	c, err := mark.Open(filepath.Join(model, "tokenizer.json"), filepath.Join(model, "tokenizer_config.json"))
	if err != nil {
		return err
	}
	if c == nil {
		return fmt.Errorf("%s has no model.cochain.safetensors", model)
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".txt") {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	if len(names) == 0 {
		return fmt.Errorf("%s has no .txt files", dir)
	}
	var chains []mark.Chain
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		chains = append(chains, c.Chain(name, string(b)))
	}
	return mark.WritePNG(out, chains)
}
