package hodge

import "testing"

func TestDistDropsFarTail(t *testing.T) {
	lg := make([]float32, 8)
	for i := range lg {
		lg[i] = -100
	}
	lg[3] = 0
	lg[5] = -11
	lg[6] = -12
	lg[7] = -13
	got := map[int]bool{}
	for _, c := range dist(lg, 0.6) {
		got[c.id] = true
	}
	if !got[3] || !got[5] || !got[6] || got[7] || got[0] {
		t.Fatalf("window %v", got)
	}
}

func TestRepeatPenaltyArgmax(t *testing.T) {
	lg := []float32{0, 5, 4.9}
	id := draw(lg, []int{1}, 0, 0.9, nil, nil, nil)
	if id != 2 {
		t.Fatalf("id %d logits %v", id, lg)
	}
}

func TestPenaltySkipsStop(t *testing.T) {
	lg := []float32{0, 5, 4}
	id := draw(lg, []int{1}, 0, 0.9, func(id int) bool { return id == 1 }, nil, nil)
	if id != 1 {
		t.Fatalf("id %d logits %v", id, lg)
	}
}
