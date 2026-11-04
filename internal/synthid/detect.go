package synthid

import (
	"fmt"
	"math"
)

// Config is the watermarking configuration the frequentist detector consumes.
// Key is the master secret, an integer in [0, 2^63). Threefry turns it into
// two 32-bit words, then each context of ngram_len-1 tokens is folded in.
// The g-bits for a token are taken from that context key. Depth is how many
// of those bits are scored. There is no built-in key. Ngram length, depth,
// and context history default to 5, 30, and 1024 when unset.
type Config struct {
	NgramLen       int
	Key            int64
	Depth          int
	ContextHistory int
	EOS            []int
}

func (c Config) norm() (Config, error) {
	if c.Key < 0 || uint64(c.Key) >= 1<<63 {
		return c, fmt.Errorf("synthid: key %d is outside [0, 2^63)", c.Key)
	}
	if c.NgramLen == 0 {
		c.NgramLen = 5
	}
	if c.NgramLen < 2 {
		return c, fmt.Errorf("synthid: ngram_len %d", c.NgramLen)
	}
	if c.Depth == 0 {
		c.Depth = 30
	}
	if c.Depth < 1 {
		return c, fmt.Errorf("synthid: depth %d", c.Depth)
	}
	if c.ContextHistory == 0 {
		c.ContextHistory = 1024
	}
	if c.ContextHistory < 1 {
		return c, fmt.Errorf("synthid: context history %d", c.ContextHistory)
	}
	return c, nil
}

// PValue is the one-sided binomial tail P(X >= ones) for the unmasked
// g-values of ids. A sequence with nothing to score returns 1.
// ids are generated token ids with the prompt already removed.
func PValue(ids []int, cfg Config) (float64, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return 0, err
	}
	ctxLen := cfg.NgramLen - 1
	nctx := len(ids) - ctxLen
	if nctx <= 0 {
		return 1, nil
	}
	eos := map[int]bool{}
	for _, id := range cfg.EOS {
		eos[id] = true
	}
	cut := len(ids)
	for i, id := range ids {
		if eos[id] {
			cut = i
			break
		}
	}
	base1, base2 := seed(uint64(cfg.Key))
	history := make([]uint64, cfg.ContextHistory)
	ones, trials := 0, 0
	for i := 0; i < nctx; i++ {
		ctx := make([]int64, ctxLen)
		for k := 0; k < ctxLen; k++ {
			ctx[k] = int64(ids[i+k])
		}
		k1, k2 := foldAll(base1, base2, ctx)
		packed := uint64(k1)<<32 | uint64(k2)
		repeated := false
		for _, h := range history {
			if h == packed {
				repeated = true
				break
			}
		}
		copy(history[1:], history[:len(history)-1])
		history[0] = packed
		end := i + ctxLen
		if end >= cut || repeated {
			continue
		}
		for _, b := range gBits(k1, k2, uint32(ids[end]), cfg.Depth) {
			trials++
			ones += b
		}
	}
	return binomSF(ones, trials), nil
}

// ExpectedMeanG is the uniform-LM mean g-value for one tournament layer.
func ExpectedMeanG(vocab, leaves int) (float64, error) {
	switch leaves {
	case 2:
		return 0.5 + 0.25*(1-1/float64(vocab)), nil
	case 3:
		return 7.0/8.0 - 3.0/(8.0*float64(vocab)), nil
	default:
		return 0, fmt.Errorf("synthid: leaves %d", leaves)
	}
}

// binomSF is P(X >= ones) for X ~ Binomial(trials, 1/2).
func binomSF(ones, trials int) float64 {
	if trials <= 0 || ones <= 0 {
		return 1
	}
	if ones > trials {
		return 0
	}
	logHalf := float64(trials) * math.Ln2
	sum := 0.0
	for i := ones; i <= trials; i++ {
		logC := lnGamma(trials+1) - lnGamma(i+1) - lnGamma(trials-i+1)
		sum += math.Exp(logC - logHalf)
	}
	if sum > 1 {
		return 1
	}
	return sum
}

func lnGamma(n int) float64 {
	g, _ := math.Lgamma(float64(n))
	return g
}
