package mark

import "testing"

func TestKeyFromSigns(t *testing.T) {
	chains := []Chain{
		{Name: "a", Centroid: []float64{1, -1, 1, -1}},
		{Name: "b", Centroid: []float64{2, -3, -1, -4}},
	}
	got, err := KeyFrom(chains)
	if err != nil {
		t.Fatal(err)
	}
	// Cell 0 plus, cell 1 minus, cell 2 disagrees, cell 3 minus.
	// Low bits first: 1 | (2<<2) | (2<<6) = 137 = 0x89.
	if got.Key != 1 || got.Mask != "+-.-" || got.Code != "89" || got.Agreed != 3 {
		t.Fatalf("key %d mask %s code %s agreed %d", got.Key, got.Mask, got.Code, got.Agreed)
	}
}
