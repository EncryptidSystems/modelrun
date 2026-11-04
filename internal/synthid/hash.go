// Package synthid is the frequentist half of the public SynthID Text detector.
// Threefry 2x32 and the g-value bits follow
// google-deepmind/synthid-text src/synthid_text, with no torch and no jax.
// The Bayesian trainer and the generation mixin are not ported.
package synthid

const skeinKSParity uint32 = 0x1BD11BDA

func rotl(x uint32, d int) uint32 {
	return (x << uint(d)) | (x >> uint(32-d))
}

func mix(v0, v1 uint32, rot int) (uint32, uint32) {
	v0 += v1
	v1 = rotl(v1, rot)
	v1 = v0 ^ v1
	return v0, v1
}

func inject(v0, v1, a, b uint32, n uint32) (uint32, uint32) {
	return v0 + a, v1 + b + n
}

// threefry is the 20-round Threefry 2x32 block from hashing_function.py.
func threefry(key1, key2, x1, x2 uint32) (uint32, uint32) {
	ks0, ks1 := key1, key2
	ks2 := key1 ^ key2 ^ skeinKSParity
	x1 += ks0
	x2 += ks1
	r0 := [4]int{13, 15, 26, 6}
	r1 := [4]int{17, 29, 16, 24}
	group := func(rots [4]int, a, b, n uint32) {
		for _, r := range rots {
			x1, x2 = mix(x1, x2, r)
		}
		x1, x2 = inject(x1, x2, a, b, n)
	}
	group(r0, ks1, ks2, 1)
	group(r1, ks2, ks0, 2)
	group(r0, ks0, ks1, 3)
	group(r1, ks1, ks2, 4)
	group(r0, ks2, ks0, 5)
	return x1, x2
}

func seed(v uint64) (uint32, uint32) {
	return uint32(v >> 32), uint32(v)
}

func fold(key1, key2 uint32, data int64) (uint32, uint32) {
	s1, s2 := seed(uint64(data))
	return threefry(key1, key2, s1, s2)
}

func foldAll(key1, key2 uint32, data []int64) (uint32, uint32) {
	for _, d := range data {
		key1, key2 = fold(key1, key2, d)
	}
	return key1, key2
}

// gBits is compute_g_values_boolean for one context key and one token.
func gBits(key1, key2 uint32, token uint32, depth int) []int {
	if depth < 1 {
		return nil
	}
	blocks := (depth + 63) / 64
	packed := make([]uint64, blocks)
	for b := 0; b < blocks; b++ {
		o1, o2 := threefry(key1, key2, token, uint32(b))
		packed[b] = uint64(o1)<<32 | uint64(o2)
	}
	out := make([]int, depth)
	for d := 0; d < depth; d++ {
		out[d] = int((packed[d/64] >> uint(d%64)) & 1)
	}
	return out
}
