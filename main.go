package main

import (
	"math"
	"os"

	"image"
	"image/color"
	"image/draw"
	"image/png"

	"flag"

	"path"

	"github.com/gonum/matrix/mat64"
	log "github.com/sirupsen/logrus"
	"golang.org/x/image/tiff"
)

var (
	i0      = flag.Int("i0", 140, "")
	j0      = flag.Int("j0", 60, "")
	k       = flag.Int("k", 3, "")
	s       = flag.Int("s", 3, "")
	rc      = flag.Float64("r", 0.7, "coeff to apply. For method1")
	metrics = flag.Int("metrics", 1, "1 - sum .^2\n2 - min .\n3 - sum .")
	method  = flag.Int("method", 1, "1,2,3,4,5")
)

func main() {
	flag.Parse()

	process()
}

func Error(err error) error {
	if err != nil {
		log.Error(err)
	}

	return err
}

func openImage(path string) (image.Image, error) {
	fin, err := os.Open(path)
	if err != nil {
		return nil, Error(err)
	}
	defer fin.Close()
	img, err := tiff.Decode(fin)

	return img, Error(err)
}

func saveImage(img image.Image, Path string) error {
	os.MkdirAll("./dist/", 0755)
	fout, err := os.Create(path.Join("./dist/", Path))
	if err != nil {
		return Error(err)
	}
	defer fout.Close()

	return Error(png.Encode(fout, img))
}

func convert(img image.Image) ([][]int32, [][]int32, [][]int32) {
	var (
		redImg   [][]int32
		greenImg [][]int32
		blueImg  [][]int32
	)
	size := img.Bounds().Max

	redImg = make([][]int32, size.X)
	greenImg = make([][]int32, size.X)
	blueImg = make([][]int32, size.X)
	for i := 0; i < size.X; i++ {
		redImg[i] = make([]int32, size.Y)
		greenImg[i] = make([]int32, size.Y)
		blueImg[i] = make([]int32, size.Y)
		for j := 0; j < size.Y; j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			r /= 257
			g /= 257
			b /= 257
			//x := int64((r << 16) + (g << 8) + b) // 24 bits
			//x := int64(b) // 24 bits

			redImg[i][j] = int32(r)
			greenImg[i][j] = int32(g)
			blueImg[i][j] = int32(b)
		}
	}

	return redImg, greenImg, blueImg
}

func fragment(img [][]int32, i0, j0, k, s int) [][]int32 {
	var u = make([][]int32, 2*k+1)

	for i := 0; i <= k; i++ {
		u[k+i] = img[i0+i][j0-s : j0+s+1]
		u[k-i] = img[i0-i][j0-s : j0+s+1]
	}

	return u
}

func multiFragment(i0, j0, k, s int, img ...[][]int32) [][][]int32 {
	var ans = make([][][]int32, len(img))
	for i, img := range img {
		ans[i] = fragment(img, i0, j0, k, s)
	}

	return ans
}

func midPoint(u [][]int32) int32 {
	var um int32

	for _, arr := range u {
		for _, a := range arr {
			um += a
		}
	}

	return um / int32(len(u)*len(u[0]))
}

func sigma(u [][]int32, midPoint int32) int32 {
	var um int32

	for _, arr := range u {
		for _, a := range arr {
			um += (a - midPoint) * (a - midPoint)
		}
	}

	return int32(math.Sqrt(float64(um) / float64(len(u)*len(u[0]))))
}

func F(u, v [][]int32) float64 {
	um, vm := midPoint(u), midPoint(v)

	var s1, s2, s3 int32

	for i := range u {
		for j := range u[i] {
			s1 += (u[i][j] - um) * (v[i][j] - vm)
			s2 += (u[i][j] - um) * (u[i][j] - um)
			s3 += (v[i][j] - vm) * (v[i][j] - vm)
		}
	}

	return float64(s1) / math.Sqrt(float64(s2)*float64(s3))
}

func method1(img [][]int32, i0, j0, k, s int) [][]float64 {
	var r = make([][]float64, len(img)-k-k)

	u := fragment(img, i0, j0, k, s)

	for i := k; i < len(img)-k; i++ {
		r[i-k] = make([]float64, len(img[i])-s-s)
		for j := s; j < len(img[i])-s; j++ {
			v := fragment(img, i, j, k, s)

			r[i-k][j-s] = 0.5 * (F(u, v) + 1)
		}
	}

	return r
}

func method2(fi [][][]int32, g [][][]int32) [][]float64 {
	var (
		l   = len(fi)
		mFi = make([]int32, l)
	)
	for i := range mFi {
		mFi[i] = midPoint(fi[i])
	}

	var data = make([]float64, l*l)
	var cov = make([][]float64, l)

	for k := range cov {
		cov[k] = data[k*l : (k+1)*l]
		for s := range cov {
			var sum int32
			for i := range fi {
				for j := range fi[i] {
					sum += (fi[k][i][j] - mFi[k]) * (fi[s][i][j] - mFi[s])
				}
			}

			cov[k][s] = float64(sum) / float64(len(fi)*len(fi[0]))
		}
	}

	ic := mat64.NewDense(l, l, data)
	for i := 0; i < l; i++ {
		ic.Set(i, i, ic.At(i, i)+1)
	}
	ic.Inverse(ic)

	var r = make([][]float64, len(g[0]))

	for i := range r {
		r[i] = make([]float64, len(g[0][i]))
		for j := range r[i] {
			var y = make([]float64, l)

			for k := range y {
				y[k] = float64(g[k][i][j] - mFi[k])
			}

			ans := &mat64.Dense{}
			ans2 := &mat64.Dense{}

			ans.Mul(mat64.NewDense(1, l, y), ic)
			ans2.Mul(ans, mat64.NewDense(l, 1, y))

			r[i][j] = ans2.At(0, 0)
		}
	}

	return r
}

func method3(g [][]int32, k int) [][]int32 {
	m := midPoint(g)
	s := sigma(g, m)

	l, r := m-2*s, m+2*s
	kk := int32(1) << uint(k)

	var dg = make([][]int32, len(g))
	for i := range g {
		dg[i] = make([]int32, len(g[i]))

		for j, g := range g[i] {
			if g < l {
				dg[i][j] = 1
			} else if g >= r {
				dg[i][j] = kk
			} else {
				dg[i][j] = (g-m+2*s)*(kk-2)/4/s + 2
			}
		}
	}

	return dg
}

func method4(g [][]int32, k int, s int) ([][]float64, [][]float64, [][]float64) {
	dg := method3(g, k)

	var (
		S = make([][]float64, len(dg)-s-s)
		E = make([][]float64, len(dg)-s-s)
		W = make([][]float64, len(dg)-s-s)
	)

	is := 1 / float64((s+1)*(s+1))

	for i := s; i < len(dg)-s; i++ {
		S[i-s] = make([]float64, len(dg[i])-s-s)
		E[i-s] = make([]float64, len(dg[i])-s-s)
		W[i-s] = make([]float64, len(dg[i])-s-s)

		for j := s; j < len(dg[i])-s; j++ {
			v := fragment(dg, i, j, s, s)
			var p = make([]float64, 1<<uint(k))
			for _, t := range v {
				for _, t := range t {
					p[t-1]++
				}
			}
			var ss, ee, ww float64

			m := midPoint(v)

			for i := range p {
				p[i] = p[i] * is

				if p[i] > 0.000001 {
					ss -= p[i] * math.Log2(p[i])
				}
				ee += p[i] * p[i]
				ww += float64((i+1-int(m))*(i+1-int(m))) * p[i]
			}

			//fmt.Println(ss)

			S[i-s][j-s] = ss
			E[i-s][j-s] = ee
			W[i-s][j-s] = ww
		}
	}

	return S, E, W
}

func method5(g1 [][]int32, g2 [][]int32, k int, s int) ([][]float64, [][]float64, [][]float64) {
	dg1 := method3(g1, k)
	dg2 := method3(g2, k)

	var (
		S = make([][]float64, len(dg1)-s-s)
		E = make([][]float64, len(dg1)-s-s)
		W = make([][]float64, len(dg1)-s-s)
	)

	is := 1 / float64((s+1)*(s+1)*(s+1)*(s+1))

	var p = make([][]float64, 1<<uint(k))
	for i := range p {
		p[i] = make([]float64, 1<<uint(k))
	}

	for i := s; i < len(dg1)-s; i++ {
		S[i-s] = make([]float64, len(dg1[i])-s-s)
		E[i-s] = make([]float64, len(dg1[i])-s-s)
		W[i-s] = make([]float64, len(dg1[i])-s-s)

		for j := s; j < len(dg1[i])-s; j++ {
			u := fragment(dg1, i, j, s, s)
			v := fragment(dg2, i, j, s, s)

			for k := range p {
				for t := range p[k] {
					p[k][t] = 0
				}
			}

			for _, t1 := range u {
				for _, t1 := range t1 {

					for _, t2 := range v {
						for _, t2 := range t2 {
							p[t1-1][t2-1]++
						}
					}
				}
			}

			var ss, ee, ww float64

			var m1, m2 float64

			for i := range p {
				for j := range p[i] {
					p[i][j] = p[i][j] * is

					if p[i][j] > 0.0000001 {
						ss -= p[i][j] * math.Log2(p[i][j])
					}
					ee += p[i][j] * p[i][j]
					m1 += float64(i) * p[i][j]
					m2 += float64(j) * p[i][j]
				}
			}
			for i := range p {
				for j := range p[i] {
					ww += (float64(i+1) - m1) * (float64(j+1) - m2) * p[i][j]
				}
			}

			S[i-s][j-s] = ss
			E[i-s][j-s] = ee
			W[i-s][j-s] = ww
		}
	}

	return S, E, W
}

// HLine draws a horizontal line
func HLine(img image.Image, x1, y, x2 int) {
	dimg := img.(draw.Image)
	for ; x1 <= x2; x1++ {
		dimg.Set(x1, y, color.RGBA{A: 255, R: 255})
	}
}

// VLine draws a veritcal line
func VLine(img image.Image, x, y1, y2 int) {
	dimg := img.(draw.Image)
	for ; y1 <= y2; y1++ {
		dimg.Set(x, y1, color.RGBA{A: 255, R: 255})
	}
}

// Rect draws a rectangle utilizing HLine() and VLine()
func Rect(img image.Image, x1, y1, x2, y2 int) {
	HLine(img, x1, y1, x2)
	HLine(img, x1, y2, x2)
	VLine(img, x1, y1, y2)
	VLine(img, x2, y1, y2)
}

func saveMethod1(img image.Image, i0, j0, k, s int) {
	redImg, greenImg, blueImg := convert(img)

	//dt:=time.Now()
	rr := method1(redImg, i0, j0, k, s)
	//fmt.Println(time.Now().Sub(dt))
	rg := method1(greenImg, i0, j0, k, s)
	rb := method1(blueImg, j0, j0, k, s)

	//wr, wg, wb := 20., 70., 40.
	rk := *rc

	min := func(a, b, c float64) float64 {
		if b < a {
			a = b
		}
		if c < a {
			a = c
		}
		return a
	}

	for i := range rr {
		for j := range rr[i] {

			var r float64
			switch *metrics {
			case 1:
				r = (rr[i][j]*rr[i][j] + rg[i][j]*rg[i][j] + rb[i][j]*rb[i][j]) * 0.3333
			case 2:
				r = min(rr[i][j], rg[i][j], rb[i][j])
			case 3:
				r = (rr[i][j] + rg[i][j] + rb[i][j]) * 0.3333
			}

			//r := (rr[i][j]*wr + rg[i][j]*wg + rb[i][j]*wb) / (wr + wg + wb)
			//r = rr[i][j] * rg[i][j] * rb[i][j]

			if r >= rk {
				Rect(img, i-1, j-1, i+2*k-1, j+2*s-1)
			}
		}
	}

	saveImage(img, "method1.png")
}

func saveMethod2(img image.Image, i0, j0, k, s int) {
	//dt := time.Now()
	redImg, greenImg, blueImg := convert(img)

	dist := method2(multiFragment(i0, j0, k, s, redImg, greenImg, blueImg), [][][]int32{redImg, greenImg, blueImg})
	//fmt.Println(time.Now().Sub(dt))

	mMin, mMax := minMax(dist)
	saveImage(makeImage(dist, mMin, mMax), "method2.png")

}

func saveMethod3(img image.Image, i0, j0, k, s int) {
	redImg, greenImg, blueImg := convert(img)

	dr := method3(redImg, 8)
	dg := method3(greenImg, 8)
	db := method3(blueImg, 8)

	rMin, rMax := minMaxDisc(dr)
	gMin, gMax := minMaxDisc(dg)
	bMin, bMax := minMaxDisc(db)

	img2 := makeColorImageDisc(dr, dg, db, rMin, rMax, gMin, gMax, bMin, bMax)

	saveImage(img2, "method3.png")
}

func saveMethod4(img image.Image, i0, j0, k, s int) {
	redImg, greenImg, blueImg := convert(img)

	for i, colImg := range [][][]int32{redImg, greenImg, blueImg} {
		S, E, W := method4(colImg, k, s)

		mMin, mMax := minMax(S)
		img4 := makeImage(S, mMin, mMax)
		saveImage(img4, "method4-S-"+cColor(i)+".png")

		mMin, mMax = minMax(E)
		img4 = makeImage(E, mMin, mMax)
		saveImage(img4, "method4-E-"+cColor(i)+".png")

		mMin, mMax = minMax(W)
		img4 = makeImage(W, mMin, mMax)
		saveImage(img4, "method4-W-"+cColor(i)+".png")
	}
}

func saveMethod5(img image.Image, i0, j0, k, s int) {
	redImg, greenImg, blueImg := convert(img)

	for i, colImg1 := range [][][]int32{redImg, greenImg, blueImg} {
		for j, colImg2 := range [][][]int32{redImg, greenImg, blueImg} {
			if j <= i {
				continue
			}

			S, E, W := method5(colImg1, colImg2, 4, 6)

			mMin, mMax := minMax(S)
			img5 := makeImage(S, mMin, mMax)
			saveImage(img5, "method5-S-"+cColor(i)+"-"+cColor(j)+".png")

			mMin, mMax = minMax(E)
			img5 = makeImage(E, mMin, mMax)
			saveImage(img5, "method5-E-"+cColor(i)+"-"+cColor(j)+".png")

			mMin, mMax = minMax(W)
			img5 = makeImage(W, mMin, mMax)
			saveImage(img5, "method5-5-"+cColor(i)+"-"+cColor(j)+".png")
		}
	}
}

func process() error {
	img, err := openImage("./Karta2.tif")
	if err != nil {
		return err
	}

	i0, j0 := *i0, *j0
	k, s := *k, *s

	switch *method {
	case 1:
		saveMethod1(img, i0, j0, k, s)
	case 2:
		saveMethod2(img, i0, j0, k, s)
	case 3:
		saveMethod3(img, i0, j0, k, s)
	case 4:
		saveMethod4(img, i0, j0, k, s)
	case 5:
		saveMethod5(img, i0, j0, k, s)
	}

	return nil
}

func makeImage(g [][]float64, min, max float64) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, len(g), len(g[0])))

	for i := range g {
		for j, c := range g[i] {
			cc := uint8((c - min) / (max - min) * 255)
			img.Set(i, j, color.RGBA{cc, cc, cc, 255})
		}
	}

	return img
}

func makeColorImage(r, g, b [][]float64, rMin, rMax, gMin, gMax, bMin, bMax float64) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, len(g), len(g[0])))

	for i := range r {
		for j := range r[i] {
			cr := uint8((r[i][j] - rMin) / (rMax - rMin) * 255)
			cg := uint8((g[i][j] - gMin) / (gMax - gMin) * 255)
			cb := uint8((b[i][j] - bMin) / (bMax - bMin) * 255)
			img.Set(i, j, color.RGBA{cr, cg, cb, 255})
		}
	}

	return img
}

func makeColorImageDisc(r, g, b [][]int32, rMin, rMax, gMin, gMax, bMin, bMax int32) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, len(g), len(g[0])))

	for i := range r {
		for j := range r[i] {
			cr := uint8((r[i][j] - rMin) * 255 / (rMax - rMin))
			cg := uint8((g[i][j] - gMin) * 255 / (gMax - gMin))
			cb := uint8((b[i][j] - bMin) * 255 / (bMax - bMin))
			img.Set(i, j, color.RGBA{cr, cg, cb, 255})
		}
	}

	return img
}

func minMax(g [][]float64) (min, max float64) {
	min = g[0][0]
	max = min

	for i := range g {
		for _, v := range g[i] {
			if v < min {
				min = v
			}

			if v > max {
				max = v
			}
		}
	}

	return
}

func minMaxDisc(g [][]int32) (min, max int32) {
	min = g[0][0]
	max = min

	for i := range g {
		for _, v := range g[i] {
			if v < min {
				min = v
			}

			if v > max {
				max = v
			}
		}
	}

	return
}

func cColor(color int) string {
	switch color {
	case 0:
		return "red"
	case 1:
		return "green"
	case 2:
		return "blue"
	}

	return "unknown"
}
