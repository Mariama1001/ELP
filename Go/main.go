package main

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/fogleman/gg"
)

func convolve(image [][]float64, kernel [][]float64) [][]float64 {
	height := len(image)
	width := len(image[0])
	kHeight := len(kernel)
	kWidth := len(kernel[0])

	padHeight := kHeight / 2
	padWidth := kWidth / 2

	paddedImage := make([][]float64, height+2*padHeight)
	for i := range paddedImage {
		paddedImage[i] = make([]float64, width+2*padWidth)
	}

	// Pad the image with zeros
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			paddedImage[i+padHeight][j+padWidth] = image[i][j]
		}
	}

	// Initialize the result image
	result := make([][]float64, height)
	for i := range result {
		result[i] = make([]float64, width)
	}

	// Perform convolution
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			sum := 0.0
			for ii := 0; ii < kHeight; ii++ {
				for jj := 0; jj < kWidth; jj++ {
					sum += paddedImage[i+ii][j+jj] * kernel[ii][jj]
				}
			}
			result[i][j] = sum
		}
	}

	return result
}

func main() {
	// Load an image
	img, err := gg.LoadImage("rasputin.jpeg")
	if err != nil {
		fmt.Println("Error loading image:", err)
		return
	}

	// Convert the image to grayscale if it's a color image
	if img.Bounds().Dx() != 0 && img.Bounds().Dy() != 0 {
		dc := gg.NewContext(img.Bounds().Dx(), img.Bounds().Dy())
		dc.DrawImage(img, 0, 0)
		img = dc.Image()
	}

	// Convert the grayscale image to a 2D float64 array
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	imageData := make([][]float64, height)
	for i := range imageData {
		imageData[i] = make([]float64, width)
	}

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, _, _, _ := img.At(x, y).RGBA()
			imageData[y][x] = float64(r) / 65535.0
		}
	}

	// Define the Sobel filter
	sobelX := [][]float64{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}

	sobelY := [][]float64{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1},
	}

	// Perform convolution for both X and Y directions
	gradientX := convolve(imageData, sobelX)
	gradientY := convolve(imageData, sobelY)

	// Combine the results to get the final edge-detected image
	edges := make([][]float64, height)
	for i := range edges {
		edges[i] = make([]float64, width)
	}

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			edges[i][j] = math.Sqrt(gradientX[i][j]*gradientX[i][j] + gradientY[i][j]*gradientY[i][j])
		}
	}

	// Normalize the pixel values to the range [0, 255]
	minValue := edges[0][0]
	maxValue := edges[0][0]
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			if edges[i][j] < minValue {
				minValue = edges[i][j]
			}
			if edges[i][j] > maxValue {
				maxValue = edges[i][j]
			}
		}
	}

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			edges[i][j] = 255 * (edges[i][j] - minValue) / (maxValue - minValue)
		}
	}

	// Convert the 2D float64 array back to a grayscale image
	edgeImg := image.NewGray(img.Bounds())
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			edgeImg.Set(x, y, color.Gray{uint8(edges[y][x])})
		}
	}

	// Save or display the original and edge-detected images
	if err := gg.SavePNG("original_image.png", img); err != nil {
		fmt.Println("Error saving original image:", err)
		return
	}

	if err := gg.SavePNG("edge_detected_image.png", edgeImg); err != nil {
		fmt.Println("Error saving edge-detected image:", err)
		return
	}
}
