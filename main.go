package main

import (
	"image"
	"image/png"
	"os"

	"math"

	"fmt"

	"image/draw"

	"image/color"

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

func convert(img image.Image) [][]int64 {
	var numImg [][]int64
	size := img.Bounds().Max

	numImg = make([][]int64, size.X)
	for i := 0; i < size.X; i++ {
		numImg[i] = make([]int64, size.Y)
		for j := 0; j < size.Y; j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			r /= 257
			g /= 257
			b /= 257
			x := int64((r << 16) + (g << 8) + b) // 24 bits

			numImg[i][j] = x
		}
	}

	return numImg
}

func fragment(img [][]int64, i0, j0, k, s int) [][]int64 {
	var u = make([][]int64, 2*k+1)

	for i := 0; i <= k; i++ {
		u[k+i] = img[i0+i][j0-s : j0+s+1]
		u[k-i] = img[i0-i][j0-s : j0+s+1]
	}

	return u
}

func midPoint(u [][]int64) int64 {
	var um int64

	for _, arr := range u {
		for _, a := range arr {
			um += a
		}
	}

	return um / int64(len(u)*len(u[0]))
}

func F(u, v [][]int64) float64 {
	um, vm := midPoint(u), midPoint(v)

	var s1, s2, s3 int64

	for i := range u {
		for j := range u[i] {
			s1 += (u[i][j] - um) * (v[i][j] - vm)
			s2 += (u[i][j] - um) * (u[i][j] - um)
			s3 += (v[i][j] - vm) * (v[i][j] - vm)
		}
	}

	return float64(s1) / math.Sqrt(float64(s2)*float64(s3))
}

func method1(img [][]int64, i0, j0, k, s int) [][]float64 {
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

	imgNum := convert(img)
	r := method1(imgNum, 123, 50, k, s)

	for i := range r {
		for j := range r[i] {
			if r[i][j] > 0.65 || r[i][j] < -0.65 {
				fmt.Println(i, j, r[i][j])

				Rect(img, i-k, j-s, i+k, j+s)
			}
		}
	}

	if err := saveImage(img, "./Karta2.png"); err != nil {
		return err
	}

	return nil
}
