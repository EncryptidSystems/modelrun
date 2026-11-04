package hodge

import (
	"math"
	"math/rand"
	"sort"
)

// The sampler is hkruntime's. dist softmaxes only the tokens within 20
// temperatures of the best and renormalizes that set. penalize is the
// repetition penalty on this answer, and min-p drops the long tail.
const (
	minP       = 0.1
	repeatPen  = 1.08
	repeatLast = 256
)

type cand struct {
	id int
	p  float64
}

// dist is softmax over the tokens within 20 temperatures of the best.
// temp <= 0 builds the distribution at temperature 1; its first entry is the argmax.
func dist(lg []float32, temp float64) []cand {
	if len(lg) == 0 {
		return nil
	}
	if temp <= 0 {
		temp = 1
	}
	best := 0
	bestV := math.Inf(-1)
	for i, v := range lg {
		fv := float64(v)
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			continue
		}
		if fv > bestV {
			best = i
			bestV = fv
		}
	}
	if math.IsInf(bestV, -1) {
		return []cand{{id: 0, p: 1}}
	}
	floor := bestV - 20*temp
	cs := make([]cand, 0, 64)
	sum := 0.0
	for i, v := range lg {
		fv := float64(v)
		if math.IsNaN(fv) || math.IsInf(fv, 0) || fv < floor {
			continue
		}
		p := math.Exp((fv - bestV) / temp)
		cs = append(cs, cand{id: i, p: p})
		sum += p
	}
	if sum == 0 || len(cs) == 0 {
		return []cand{{id: best, p: 1}}
	}
	for i := range cs {
		cs[i].p /= sum
	}
	sort.SliceStable(cs, func(i, j int) bool { return cs[i].p > cs[j].p })
	return cs
}

// penalize applies the repetition penalty once per distinct id in recent.
// A positive logit is divided by pen. A negative one is multiplied.
// keep(id) true leaves that id alone, so an end token is never held open.
func penalize(lg []float32, recent []int, pen float64, keep func(int) bool) {
	if pen <= 1 {
		return
	}
	for i, id := range recent {
		if id < 0 || id >= len(lg) || (keep != nil && keep(id)) {
			continue
		}
		seen := false
		for _, prev := range recent[:i] {
			if prev == id {
				seen = true
				break
			}
		}
		if seen {
			continue
		}
		v := float64(lg[id])
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		if v > 0 {
			v /= pen
		} else {
			v *= pen
		}
		lg[id] = float32(v)
	}
}

func remember(recent []int, id, n int) []int {
	recent = append(recent, id)
	if n > 0 && len(recent) > n {
		recent = recent[len(recent)-n:]
	}
	return recent
}

// draw penalizes, then samples. temp <= 0 is the best token after the penalty.
// end(id) true stays in the draw after min-p and top-p.
func draw(lg []float32, recent []int, temp, top float64, keep, end func(int) bool, rng *rand.Rand) int {
	penalize(lg, recent, repeatPen, keep)
	return pick(dist(lg, temp), temp, top, minP, end, rng)
}

func pick(cs []cand, temp, topP, minP float64, end func(int) bool, rng *rand.Rand) int {
	if len(cs) == 0 {
		return 0
	}
	if temp <= 0 || rng == nil {
		return cs[0].id
	}
	keep := len(cs)
	if minP > 0 && minP < 1 {
		floor := cs[0].p * minP
		for i := 0; i < keep; i++ {
			if cs[i].p < floor {
				keep = i
				break
			}
		}
	}
	if keep < 1 {
		keep = 1
	}
	if topP > 0 && topP < 1 {
		acc := 0.0
		for i := 0; i < keep; i++ {
			acc += cs[i].p
			if acc >= topP {
				keep = i + 1
				break
			}
		}
	}
	var extra []cand
	if end != nil {
		for _, c := range cs[keep:] {
			if !end(c.id) {
				continue
			}
			seen := false
			for i := 0; i < keep; i++ {
				if cs[i].id == c.id {
					seen = true
					break
				}
			}
			if !seen {
				extra = append(extra, c)
			}
		}
	}
	total := 0.0
	for i := 0; i < keep; i++ {
		total += cs[i].p
	}
	for _, c := range extra {
		total += c.p
	}
	if total == 0 {
		return cs[0].id
	}
	x := rng.Float64() * total
	for i := 0; i < keep; i++ {
		x -= cs[i].p
		if x <= 0 {
			return cs[i].id
		}
	}
	for _, c := range extra {
		x -= c.p
		if x <= 0 {
			return c.id
		}
	}
	if len(extra) > 0 {
		return extra[len(extra)-1].id
	}
	return cs[keep-1].id
}
