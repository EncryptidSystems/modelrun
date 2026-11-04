// Package tokenizers loads a Hugging Face tokenizer.json and encodes and
// decodes with it: byte-level BPE (GPT-2 byte alphabet), added special
// tokens, and the Split pre-tokenizer of the Llama 3 / cl100k family.
//
// The Split regex is implemented by hand (this regexp has no look-ahead and
// no \p classes); Load checks the file's pattern is that one and refuses any
// other rather than tokenizing differently from the model's training.
package tokenizers

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The Llama 3 / cl100k pre-tokenizer, as tokenizer.json spells it.
const llama3Split = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type special struct {
	text string
	id   int
}

// Tokenizer is a loaded tokenizer.json.
type Tokenizer struct {
	vocab        map[string]int
	tokens       []string       // id -> token text (byte-level alphabet), or a special's text
	isSpecial    []byte         // 1 for an added special token
	ranks        map[string]int // "left right" -> merge rank
	specials     []special      // longest first
	ignoreMerges bool
	gemma4       bool         // space becomes ▁, then one BPE over the whole string
	byteChar     []string     // byte -> its byte-level rune, as a string
	charByte     map[rune]int // byte-level rune -> byte
	cache        map[string][]int
}

// Size is the number of token ids (vocabulary plus added tokens).
func (t *Tokenizer) Size() int {
	return len(t.tokens)
}

// bracketToken is a control-shaped piece, <bos> or <|eot_id|> or <start_of_turn>.
func bracketToken(s string) bool {
	return len(s) >= 3 && s[0] == '<' && s[len(s)-1] == '>' && !strings.Contains(s, " ")
}

// ID is the id of token text (a special's text, or a vocabulary entry), or -1.
func (t *Tokenizer) ID(text string) int {
	for _, s := range t.specials {
		if s.text == text {
			return s.id
		}
	}
	if id, ok := t.vocab[text]; ok {
		return id
	}
	return -1
}

// IsSpecial reports whether id is an added special token.
func (t *Tokenizer) IsSpecial(id int) bool {
	return id >= 0 && id < len(t.isSpecial) && t.isSpecial[id] == 1
}

func field(m map[string]any, k string) map[string]any {
	v, _ := m[k].(map[string]any)
	return v
}

// str is m[k] when it is a string, else "".
func str(m map[string]any, k string) string {
	s, _ := m[k].(string)
	return s
}

// GPT-2's byte alphabet: printable bytes stand for themselves, the rest
// are moved to U+0100 and up.
func (t *Tokenizer) byteAlphabet() {
	t.byteChar = make([]string, 256)
	t.charByte = map[rune]int{}
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
			r = rune(256 + n)
			n++
		}
		t.byteChar[b] = string(r)
		t.charByte[r] = b
	}
}

func checkPre(pre map[string]any) error {
	if pre == nil {
		return errors.New("tokenizers: no pre_tokenizer")
	}
	steps := []any{pre}
	if str(pre, "type") == "Sequence" {
		steps, _ = pre["pretokenizers"].([]any)
	}
	split := false
	for _, s := range steps {
		m, _ := s.(map[string]any)
		switch str(m, "type") {
		case "Split":
			pat, _ := field(m, "pattern")["Regex"].(string)
			if pat != llama3Split {
				return errors.New("tokenizers: unsupported Split pattern " + pat)
			}
			split = true
		case "ByteLevel":
			if use, _ := m["use_regex"].(bool); use {
				return errors.New("tokenizers: ByteLevel with its own regex is not supported")
			}
		default:
			t, _ := m["type"].(string)
			return errors.New("tokenizers: unsupported pre_tokenizer " + t)
		}
	}
	if !split {
		return errors.New("tokenizers: pre_tokenizer has no Split")
	}
	return nil
}

// isGemma4 is the Gemma 4 / Gemini tokenizer: replace each space with ▁,
// then run BPE on the whole string, with <0xHH> for a character that is
// not itself a token.
func isGemma4(root, model map[string]any) bool {
	norm := field(root, "normalizer")
	pre := field(root, "pre_tokenizer")
	if str(norm, "type") != "Replace" || str(pre, "type") != "Split" {
		return false
	}
	if str(field(norm, "pattern"), "String") != " " || str(norm, "content") != "▁" {
		return false
	}
	if str(field(pre, "pattern"), "String") != " " || str(pre, "behavior") != "MergedWithPrevious" {
		return false
	}
	fb, _ := model["byte_fallback"].(bool)
	return fb
}

// Load reads a tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, errors.New("tokenizers: " + path + ": " + err.Error())
	}
	root, _ := v.(map[string]any)
	model := field(root, "model")
	if model == nil || str(model, "type") != "BPE" {
		return nil, errors.New("tokenizers: " + path + ": model is not BPE")
	}
	gemma4 := isGemma4(root, model)
	if !gemma4 {
		if err := checkPre(field(root, "pre_tokenizer")); err != nil {
			return nil, err
		}
	}
	t := &Tokenizer{vocab: map[string]int{}, ranks: map[string]int{}, cache: map[string][]int{}, gemma4: gemma4}
	t.byteAlphabet()
	t.ignoreMerges, _ = model["ignore_merges"].(bool)
	vocab := field(model, "vocab")
	size := 0
	for tok, idv := range vocab {
		id := int(idv.(float64))
		t.vocab[tok] = id
		if id+1 > size {
			size = id + 1
		}
	}
	added, _ := root["added_tokens"].([]any)
	for _, a := range added {
		m, _ := a.(map[string]any)
		id := int(m["id"].(float64))
		if id+1 > size {
			size = id + 1
		}
	}
	t.tokens = make([]string, size)
	t.isSpecial = make([]byte, size)
	ids := t.vocab
	for tok, id := range ids {
		t.tokens[id] = tok
	}
	for _, a := range added {
		m, _ := a.(map[string]any)
		id := int(m["id"].(float64))
		text, _ := m["content"].(string)
		t.tokens[id] = text
		sp, _ := m["special"].(bool)
		if sp {
			t.isSpecial[id] = 1
		}
		// User-defined pieces such as <start_of_turn> are added with special false.
		// They are still one id. A piece that is not wrapped in <> stays on the BPE path,
		// so an added newline does not steal the newline from the merges.
		if text != "" && (sp || bracketToken(text)) {
			t.specials = append(t.specials, special{text: text, id: id})
		}
	}
	// Longest first, so a special that prefixes another does not win.
	sp := t.specials
	for i := 1; i < len(sp); i++ {
		for j := i; j > 0 && len(sp[j].text) > len(sp[j-1].text); j-- {
			sp[j], sp[j-1] = sp[j-1], sp[j]
		}
	}
	merges, _ := model["merges"].([]any)
	for i, mv := range merges {
		if s, ok := mv.(string); ok {
			t.ranks[s] = i
		} else if pair, ok := mv.([]any); ok && len(pair) == 2 {
			a, _ := pair[0].(string)
			b, _ := pair[1].(string)
			t.ranks[a+" "+b] = i
		}
	}
	return t, nil
}

// ---- the Llama 3 Split, by hand

func isL(r rune) bool  { return unicode.IsLetter(r) }
func isN(r rune) bool  { return unicode.IsNumber(r) }
func isS(r rune) bool  { return unicode.IsSpace(r) }
func isNL(r rune) bool { return r == '\r' || r == '\n' }

func lower(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + 32
	}
	return r
}

// split cuts text into the pieces the regex would match, leftmost-first
// alternation at each position.
func split(text string) []string {
	rs := []rune(text)
	n := len(rs)
	var out []string
	i := 0
	for i < n {
		j := matchAt(rs, i)
		out = append(out, string(rs[i:j]))
		i = j
	}
	return out
}

func matchAt(rs []rune, i int) int {
	n := len(rs)
	c := rs[i]
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if c == '\'' && i+1 < n {
		a := lower(rs[i+1])
		if a == 's' || a == 't' || a == 'm' || a == 'd' {
			return i + 2
		}
		if i+2 < n {
			b := lower(rs[i+2])
			if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') {
				return i + 3
			}
		}
	}
	// [^\r\n\p{L}\p{N}]?\p{L}+
	j := i
	if !isNL(c) && !isL(c) && !isN(c) && i+1 < n && isL(rs[i+1]) {
		j = i + 1
	}
	if j < n && isL(rs[j]) {
		for j < n && isL(rs[j]) {
			j++
		}
		return j
	}
	// \p{N}{1,3}
	if isN(c) {
		j = i
		for j < n && j < i+3 && isN(rs[j]) {
			j++
		}
		return j
	}
	// ' ?[^\s\p{L}\p{N}]+[\r\n]*'
	other := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }
	j = i
	if c == ' ' && i+1 < n && other(rs[i+1]) {
		j = i + 1
	}
	if j < n && other(rs[j]) {
		for j < n && other(rs[j]) {
			j++
		}
		for j < n && isNL(rs[j]) {
			j++
		}
		return j
	}
	// The rest start with white space: its maximal run is [i, end).
	end := i
	for end < n && isS(rs[end]) {
		end++
	}
	// \s*[\r\n]+ : up to the run's last line break.
	for k := end - 1; k >= i; k-- {
		if isNL(rs[k]) {
			return k + 1
		}
	}
	// \s+(?!\S) : the run, or all but its last char when text follows.
	if end == n {
		return end
	}
	if end-1 > i {
		return end - 1
	}
	// \s+
	return end
}

// ---- BPE

func (t *Tokenizer) bpe(piece string) []int {
	if ids, ok := t.cache[piece]; ok {
		return ids
	}
	var sb strings.Builder
	for k := 0; k < len(piece); k++ {
		sb.WriteString(t.byteChar[piece[k]])
	}
	word := sb.String()
	if t.ignoreMerges {
		if id, ok := t.vocab[word]; ok {
			ids := []int{id}
			t.cache[piece] = ids
			return ids
		}
	}
	var syms []string
	for _, r := range word {
		syms = append(syms, string(r))
	}
	for len(syms) > 1 {
		best := -1
		bestRank := 0
		for k := 0; k+1 < len(syms); k++ {
			if r, ok := t.ranks[syms[k]+" "+syms[k+1]]; ok && (best < 0 || r < bestRank) {
				best = k
				bestRank = r
			}
		}
		if best < 0 {
			break
		}
		a := syms[best]
		b := syms[best+1]
		var next []string
		for k := 0; k < len(syms); k++ {
			if k+1 < len(syms) && syms[k] == a && syms[k+1] == b {
				next = append(next, a+b)
				k++
			} else {
				next = append(next, syms[k])
			}
		}
		syms = next
	}
	ids := make([]int, 0, len(syms))
	for _, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		}
	}
	t.cache[piece] = ids
	return ids
}

func (t *Tokenizer) encodePlain(text string, ids []int) []int {
	if t.gemma4 {
		return append(ids, t.gemma4IDs(text)...)
	}
	pieces := split(text)
	for _, p := range pieces {
		ids = append(ids, t.bpe(p)...)
	}
	return ids
}

// gemma4IDs is text under the Gemma 4 normalizer and BPE.
func (t *Tokenizer) gemma4IDs(text string) []int {
	if ids, ok := t.cache[text]; ok {
		return ids
	}
	var syms []string
	for _, r := range strings.ReplaceAll(text, " ", "▁") {
		s := string(r)
		if _, ok := t.vocab[s]; ok {
			syms = append(syms, s)
			continue
		}
		for _, b := range []byte(s) {
			syms = append(syms, fmt.Sprintf("<0x%02X>", b))
		}
	}
	for len(syms) > 1 {
		best := -1
		bestRank := 0
		for k := 0; k+1 < len(syms); k++ {
			if r, ok := t.ranks[syms[k]+" "+syms[k+1]]; ok && (best < 0 || r < bestRank) {
				best = k
				bestRank = r
			}
		}
		if best < 0 {
			break
		}
		a, b := syms[best], syms[best+1]
		var next []string
		for k := 0; k < len(syms); k++ {
			if k+1 < len(syms) && syms[k] == a && syms[k+1] == b {
				next = append(next, a+b)
				k++
			} else {
				next = append(next, syms[k])
			}
		}
		syms = next
	}
	ids := make([]int, 0, len(syms))
	unk := t.vocab["<unk>"]
	for _, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, unk)
		}
	}
	t.cache[text] = ids
	return ids
}

// Encode is the ids of text. With specials, added special tokens written
// in the text (such as "<|eot_id|>") are their ids; without, they are text.
func (t *Tokenizer) Encode(text string, specials bool) []int {
	var ids []int
	if !specials || len(t.specials) == 0 {
		return t.encodePlain(text, ids)
	}
	start := 0
	i := 0
	for i < len(text) {
		hit := -1
		for k, s := range t.specials {
			if strings.HasPrefix(text[i:], s.text) {
				hit = k
				break
			}
		}
		if hit < 0 {
			i++
			continue
		}
		if i > start {
			ids = t.encodePlain(text[start:i], ids)
		}
		ids = append(ids, t.specials[hit].id)
		i += len(t.specials[hit].text)
		start = i
	}
	if start < len(text) {
		ids = t.encodePlain(text[start:], ids)
	}
	return ids
}

// Bytes is the text of one id as bytes: a vocabulary token's byte-level
// letters turned back into bytes, or a special's text. A token may end in
// the middle of a UTF-8 sequence; join tokens before decoding as text.
func (t *Tokenizer) Bytes(id int) []byte {
	if id < 0 || id >= len(t.tokens) {
		return nil
	}
	if t.isSpecial[id] == 1 {
		return []byte(t.tokens[id])
	}
	var out []byte
	for _, r := range t.tokens[id] {
		if b, ok := t.charByte[r]; ok {
			out = append(out, byte(b))
		} else {
			var tmp [4]byte
			n := utf8.EncodeRune(tmp[:], r)
			out = append(out, tmp[:n]...)
		}
	}
	return out
}

// Decode is the text of ids. Specials are left out unless keepSpecials.
func (t *Tokenizer) Decode(ids []int, keepSpecials bool) string {
	if t.gemma4 {
		return t.decodeGemma4(ids, keepSpecials)
	}
	var out []byte
	for _, id := range ids {
		if t.IsSpecial(id) && !keepSpecials {
			continue
		}
		out = append(out, t.Bytes(id)...)
	}
	return string(out)
}

func (t *Tokenizer) decodeGemma4(ids []int, keepSpecials bool) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.tokens) {
			continue
		}
		if t.IsSpecial(id) && !keepSpecials {
			continue
		}
		s := t.tokens[id]
		if len(s) == 6 && strings.HasPrefix(s, "<0x") && s[5] == '>' {
			n, err := strconv.ParseUint(s[3:5], 16, 8)
			if err == nil {
				b.WriteByte(byte(n))
				continue
			}
		}
		b.WriteString(s)
	}
	return strings.ReplaceAll(b.String(), "▁", " ")
}
