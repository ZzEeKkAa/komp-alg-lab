package main

import (
	"fmt"
	"math"
	"os"

	"image"
	"image/color"
	"image/draw"
	"image/png"

	"github.com/gonum/matrix/mat64"
	log "github.com/sirupsen/logrus"
	"golang.org/x/image/tiff"
)

func main() {
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

func saveImage(img image.Image, path string) error {
	fout, err := os.Create(path)
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

			r[i-k][j-s] = F(u, v)
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

				ss -= p[i] * math.Log2(p[i])
				ee += p[i] * p[i]
				ww += float64((i+1-int(m))*(i+1-int(m))) * p[i]
			}

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

					ss -= p[i][j] * math.Log2(p[i][j])
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

func process() error {
	img, err := openImage("./Karta2.tif")
	if err != nil {
		return err
	}

	k, s := 6, 10

	redImg, greenImg, blueImg := convert(img)

	dist := method2(multiFragment(123, 50, k, s, redImg, greenImg, blueImg), [][][]int32{redImg, greenImg, blueImg})

	fmt.Println(dist)

	dg := method3(redImg, 4)

	fmt.Println(dg)

	S, E, W := method4(redImg, 4, 6)

	fmt.Println(S)
	fmt.Println(E)
	fmt.Println(W)

	S, E, W = method5(redImg, greenImg, 6, s)

	fmt.Println(S)
	fmt.Println(E)
	fmt.Println(W)

	rr := method1(redImg, 123, 50, k, s)
	rg := method1(greenImg, 123, 50, k, s)
	rb := method1(blueImg, 123, 50, k, s)

	//wr, wg, wb := 20., 70., 40.
	rk := 0.65

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
			//r := (rr[i][j]*wr + rg[i][j]*wg + rb[i][j]*wb) / (wr + wg + wb)
			//r := rr[i][j] * rg[i][j] * rb[i][j]
			r := min(rr[i][j], rg[i][j], rb[i][j])

			if r > rk || r < -rk {
				fmt.Println(i, j, r)

				Rect(img, i-k, j-s, i+k, j+s)
			}
		}
	}

	if err := saveImage(img, "./Karta2.png"); err != nil {
		return err
	}

	return nil
}
