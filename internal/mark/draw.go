package mark

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"sort"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

var (
	bg    = color.RGBA{7, 9, 13, 255}
	ink   = color.RGBA{236, 231, 223, 255}
	muted = color.RGBA{142, 150, 163, 255}
	faint = color.RGBA{94, 102, 115, 255}
	warm  = color.RGBA{232, 168, 124, 255}
	cool  = color.RGBA{126, 182, 232, 255}
)

// WritePNG draws each chain's word cochain, then the shared centroid.
// Columns are cells. Copper is above the vocabulary mean, blue is below.
func WritePNG(path string, chains []Chain) error {
	if len(chains) == 0 || len(chains[0].Centroid) == 0 {
		return fmt.Errorf("no cochain to draw")
	}
	const cellW = 11
	const rowH = 2
	const left = 28
	const top = 78
	C := len(chains[0].Centroid)
	width := left + C*cellW + 120
	height := top
	for _, ch := range chains {
		height += 22 + len(ch.Anchors)*rowH + 8 + 16 + 26
	}
	height += 36 + 18*len(chains) + 22 + 36
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	fill(img, bg)
	text(img, left, 18, "Three poems on the integer cochain", ink)
	text(img, left, 38, "Each row is one word, none repeated. Columns are the 64 cells.", muted)
	text(img, left, 54, "Copper is above the vocabulary mean. Blue is below. One scale for every word.", muted)

	scale := percentile(chains, 0.98)
	if scale <= 0 {
		scale = 1
	}
	y := top
	for _, ch := range chains {
		text(img, left, y, fmt.Sprintf("%s    %d words    alignment %.3f", ch.Name, len(ch.Words), ch.Alignment), warm)
		y += 18
		for _, row := range ch.Anchors {
			for g, v := range row {
				drawCell(img, left+g*cellW, y, cellW, rowH, v, scale)
			}
			y += rowH
		}
		y += 8
		fp := maxAbs(ch.Centroid)
		if fp < 1e-9 {
			fp = 1
		}
		for g, v := range ch.Centroid {
			drawCell(img, left+g*cellW, y, cellW, 14, v, fp)
		}
		text(img, left+C*cellW+6, y+1, "centroid", faint)
		y += 16 + 26
	}

	text(img, left, y, "Centroids, one scale. The last row is the sign they share.", ink)
	y += 20
	fp := 0.0
	for _, ch := range chains {
		a := maxAbs(ch.Centroid)
		if a > fp {
			fp = a
		}
	}
	if fp < 1e-9 {
		fp = 1
	}
	for _, ch := range chains {
		for g, v := range ch.Centroid {
			drawCell(img, left+g*cellW, y, cellW, 14, v, fp)
		}
		text(img, left+C*cellW+6, y+1, ch.Name, faint)
		y += 18
	}
	agree := 0
	for g := 0; g < C; g++ {
		sign := chains[0].Centroid[g] >= 0
		same := true
		for _, ch := range chains[1:] {
			if (ch.Centroid[g] >= 0) != sign {
				same = false
				break
			}
		}
		col := faint
		if same {
			agree++
			if sign {
				col = warm
			} else {
				col = cool
			}
		}
		rect(img, left+g*cellW, y, cellW-1, 12, col)
	}
	text(img, left+C*cellW+6, y, fmt.Sprintf("%d/%d", agree, C), faint)
	y += 22
	for g := 0; g < C; g += 8 {
		text(img, left+g*cellW, y, fmt.Sprintf("%d", g), faint)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func drawCell(img *image.RGBA, x, y, w, h int, v, scale float64) {
	mag := math.Abs(v) / scale
	if mag > 1 {
		mag = 1
	}
	src := warm
	if v < 0 {
		src = cool
	}
	a := 0.16 + 0.84*mag
	if math.Abs(v) < 1e-12 {
		a = 0
	}
	rect(img, x, y, w, h, mix(bg, src, a))
}

func percentile(chains []Chain, p float64) float64 {
	var vals []float64
	for _, ch := range chains {
		for _, row := range ch.Anchors {
			for _, v := range row {
				vals = append(vals, math.Abs(v))
			}
		}
	}
	if len(vals) == 0 {
		return 1
	}
	sort.Float64s(vals)
	i := int(p * float64(len(vals)-1))
	if i < 0 {
		i = 0
	}
	if vals[i] == 0 {
		return vals[len(vals)-1]
	}
	return vals[i]
}

func maxAbs(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}

func fill(img *image.RGBA, c color.RGBA) {
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i] = c.R
		img.Pix[i+1] = c.G
		img.Pix[i+2] = c.B
		img.Pix[i+3] = c.A
	}
}

func rect(img *image.RGBA, x, y, w, h int, c color.RGBA) {
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			if image.Pt(xx, yy).In(img.Bounds()) {
				img.SetRGBA(xx, yy, c)
			}
		}
	}
}

func mix(a, b color.RGBA, t float64) color.RGBA {
	return color.RGBA{
		R: uint8(float64(a.R)*(1-t) + float64(b.R)*t),
		G: uint8(float64(a.G)*(1-t) + float64(b.G)*t),
		B: uint8(float64(a.B)*(1-t) + float64(b.B)*t),
		A: 255,
	}
}

func text(img *image.RGBA, x, y int, s string, col color.Color) {
	d := font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x, y+11),
	}
	d.DrawString(s)
}
