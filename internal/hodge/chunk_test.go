package hodge

import "testing"

func TestChunkRows(t *testing.T) {
	const row = 2560 * 2
	if got := chunkRows(row, 298648); got != 298648 {
		t.Fatalf("hodge-3.5 one chunk, got %d", got)
	}
	got := chunkRows(row, 514304)
	if got%64 != 0 || got*row > maxStorage || got >= 514304 {
		t.Fatalf("hodge-4 chunk %d", got)
	}
	if 514304%got == 0 {
		t.Fatal("split should leave a remainder or an even second chunk")
	}
}

func TestRopeLlama3(t *testing.T) {
	scale, factor, low, high, orig := ropeOf([]byte(`{"rope_type":"llama3","factor":32,"low_freq_factor":1,"high_freq_factor":4,"original_max_position_embeddings":8192}`))
	if scale != 2 || factor != 32 || low != 1 || high != 4 || orig != 8192 {
		t.Fatalf("%d %v %v %v %v", scale, factor, low, high, orig)
	}
}
