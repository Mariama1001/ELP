package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"sync"
	"time"
)



// ConvolveResult represents the result of a convolution operation
type ConvolveResult struct {
	Result [][]float64
	Index  int
}

// convolveParallel performs convolution using goroutines on each row
func convolveParallel(image [][]float64, kernel [][]float64) [][]float64 {
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

	var wg sync.WaitGroup
	resultCh := make(chan ConvolveResult, height)

	// Perform convolution using goroutines
	for i := 0; i < height; i++ {
		wg.Add(1)
		go func(row int) {
			defer wg.Done()

			startRow := row
			endRow := row + 1

			for i := startRow; i < endRow; i++ {
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

			resultCh <- ConvolveResult{Result: result[startRow:endRow], Index: row}
		}(i)
	}

	// Wait for all goroutines to finish
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Combine results from goroutines
	for convResult := range resultCh {
		startRow := convResult.Index
		copy(result[startRow], convResult.Result[0])
	}

	return result
}


func main() {

	// Load an image
	img, err := loadImage("manypixels.jpg")
	if err != nil {
		fmt.Println("Error loading image:", err)
		return
	}

	startTime := time.Now()

	// Convert the image to grayscale if it's a color image
	if isColorImage(img) {
		img = convertToGrayscale(img)
	}

	// Convert the grayscale image to a 2D float64 array
	imageData := imageToFloat64ArrayParallel(img)

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

	// Perform convolution for both X and Y directions using goroutines
	gradientX := convolveParallel(imageData, sobelX)
	gradientY := convolveParallel(imageData, sobelY)

	// Combine the results to get the final edge-detected image
	edges := make([][]float64, len(gradientX))
	for i := range edges {
		edges[i] = make([]float64, len(gradientX[0]))
	}

	for i := 0; i < len(gradientX); i++ {
		for j := 0; j < len(gradientX[0]); j++ {
			edges[i][j] = math.Sqrt(gradientX[i][j]*gradientX[i][j] + gradientY[i][j]*gradientY[i][j])
		}
	}

	// Normalize the pixel values to the range [0, 255]
	minValue := edges[0][0]
	maxValue := edges[0][0]
	for i := 0; i < len(edges); i++ {
		for j := 0; j < len(edges[0]); j++ {
			if edges[i][j] < minValue {
				minValue = edges[i][j]
			}
			if edges[i][j] > maxValue {
				maxValue = edges[i][j]
			}
		}
	}

	for i := 0; i < len(edges); i++ {
		for j := 0; j < len(edges[0]); j++ {
			edges[i][j] = 255 * (edges[i][j] - minValue) / (maxValue - minValue)
		}
	}

	// Convert the 2D float64 array back to a grayscale image
	edgeImg := float64ArrayToImageParallel(edges)

	// Save or display the original and edge-detected images
	if err := saveImage("original_image.jpg", img); err != nil {
		fmt.Println("Error saving original image:", err)
		return
	}

	if err := saveImage("edge_detected_image.jpg", edgeImg); err != nil {
		fmt.Println("Error saving edge-detected image:", err)
		return
	}

	endTime := time.Now()
	fmt.Printf("Duration with goroutines: %s\n", endTime.Sub(startTime))
}


// loadImage loads an image from file
func loadImage(filename string) (image.Image, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	return img, nil
}

// isColorImage checks if an image is in color
func isColorImage(img image.Image) bool {
	_, _, _, a := img.At(0, 0).RGBA()
	return a != 0xFFFF
}

// convertToGrayscale converts a color image to grayscale using goroutines
func convertToGrayscale(img image.Image) *image.Gray {
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	var wg sync.WaitGroup

	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		wg.Add(1)
		go func(x int) {
			defer wg.Done()
			for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
				gray.Set(x, y, img.At(x, y))
			}
		}(x)
	}

	wg.Wait()

	return gray
}

// imageToFloat64Array converts an image to a 2D float64 array using goroutines
func imageToFloat64ArrayParallel(img image.Image) [][]float64 {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	imageData := make([][]float64, height)
	var wg sync.WaitGroup

	for y := 0; y < height; y++ {
		wg.Add(1)
		go func(y int) {
			defer wg.Done()
			imageData[y] = make([]float64, width)
			for x := 0; x < width; x++ {
				r, _, _, _ := img.At(x, y).RGBA()
				imageData[y][x] = float64(r) / 65535.0
			}
		}(y)
	}

	wg.Wait()

	return imageData
}

// float64ArrayToImage converts a 2D float64 array to a grayscale image using goroutines
func float64ArrayToImageParallel(data [][]float64) *image.Gray {
	height := len(data)
	width := len(data[0])

	gray := image.NewGray(image.Rect(0, 0, width, height))
	var wg sync.WaitGroup

	for y := 0; y < height; y++ {
		wg.Add(1)
		go func(y int) {
			defer wg.Done()
			for x := 0; x < width; x++ {
				gray.SetGray(x, y, color.Gray{uint8(data[y][x])})
			}
		}(y)
	}

	wg.Wait()

	return gray
}
// saveImage saves an image to file
func saveImage(filename string, img image.Image) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	err = jpeg.Encode(file, img, nil)
	if err != nil {
		return err
	}

	return nil
}