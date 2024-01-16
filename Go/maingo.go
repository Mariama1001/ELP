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
	"log"
	"runtime"
)

// Prepare the image that is going to be treated by the edge detection filter
func PrepareImage(filename string) (image.Image) {
	
	// Load the image
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal("Error loading image:", err)
	}
	defer file.Close()

	// Decode the image so it can be treated
	img, _, err := image.Decode(file)
	if err != nil {
		log.Fatal("Error decoding image:", err)
	}

	return img
}


// ConvolveResult represents the result of a convolution operation
type ConvolveResult struct {
	Result [][]float64
	Index  int
}

var sobelX = [][]float64{
	{-1, 0, 1},
	{-2, 0, 2},
	{-1, 0, 1},
}

var sobelY = [][]float64{
	{-1, -2, -1},
	{0, 0, 0},
	{1, 2, 1},
}

// convolveParallel performs convolution using goroutines on each row
func convolveParallel(image [][]float64, kernel [][]float64, numWorkers int) [][]float64 {
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

	// Calculate the number of rows each worker should handle
	rowsPerWorker := height / numWorkers

	// Perform convolution using goroutines on each row
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			startRow := workerID * rowsPerWorker
			endRow := startRow + rowsPerWorker

			// The last worker may handle extra rows to cover any remainder
			if workerID == numWorkers-1 {
				endRow = height
			}

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

			resultCh <- ConvolveResult{Result: result[startRow:endRow], Index: workerID}
		}(w)
	}

	// Wait for all goroutines to finish
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Combine results from goroutines
	for convResult := range resultCh {
		startRow := convResult.Index * rowsPerWorker
		copy(result[startRow:], convResult.Result)
	}

	return result
}

func main() {

	// Loads and decodes the image
	img := PrepareImage("manypixels.jpg")

	numWorkers := runtime.NumCPU()
	
	startTime := time.Now()

	// Convert the image to grayscale if it's a color image
	img = convertToGrayscale(img,numWorkers)

	// Convert the grayscale image to a 2D float64 array
	imageData := imageToFloat64Array(img,numWorkers)

	
	// Perform convolution for both X and Y directions using goroutines
	gradientX := convolveParallel(imageData, sobelX, numWorkers)
	gradientY := convolveParallel(imageData, sobelY, numWorkers)	

	//startTime := time.Now()
	// Combine the results to get the final edge-detected image
	edges := combineAndNormalize(gradientX, gradientY)

	//endTime := time.Now()

	// Convert the 2D float64 array back to a grayscale image
	edgeImg := float64ArrayToImage(edges,numWorkers)


	//Save the edge detected image 
	saveImage("edge_detected_image.jpg", edgeImg)

	endTime := time.Now()
	fmt.Printf("Duration with goroutines: %s\n", endTime.Sub(startTime))
}


//convertToGrayscale converts a color image to grayscale using goroutines
func convertToGrayscale(img image.Image, numWorkers int) *image.Gray {
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	var wg sync.WaitGroup
	rowsPerWorker := (bounds.Max.Y - bounds.Min.Y) / numWorkers

	for workerID := 0; workerID < numWorkers; workerID++ {
		startRow := bounds.Min.Y + workerID*rowsPerWorker
		endRow := startRow + rowsPerWorker
		if workerID == numWorkers-1 {
			// Ensure that the last worker processes remaining rows
			endRow = bounds.Max.Y
		}

		wg.Add(1)
		go func(startRow, endRow int) {
			defer wg.Done()
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				for y := startRow; y < endRow && y < bounds.Max.Y; y++ {
					gray.Set(x, y, img.At(x, y))
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()

	return gray
}



// imageToFloat64Array converts an image to a 2D float64 array using goroutines

func imageToFloat64Array(img image.Image, numWorkers int) [][]float64 {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	imageData := make([][]float64, height)

	var wg sync.WaitGroup
	wg.Add(numWorkers) // Increment the wait group count for each worker

	rowsPerWorker := height / numWorkers

	for i := 0; i < numWorkers; i++ {
		startRow := i * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if i == numWorkers-1 {
			// Ensure that the last worker processes remaining rows
			endRow = height
		}

		go func(startRow, endRow int) {
			defer wg.Done() // Decrement the wait group count when the goroutine finishes

			for y := startRow; y < endRow; y++ {
				imageData[y] = make([]float64, width)
				for x := 0; x < width; x++ {
					r, _, _, _ := img.At(x, y).RGBA()
					imageData[y][x] = float64(r) / 65535.0
				}
			}
		}(startRow, endRow)
	}

	wg.Wait() // Wait for all goroutines to finish

	return imageData
}

func float64ArrayToImage(data [][]float64, numWorkers int) *image.Gray {
	height := len(data)
	width := len(data[0])

	gray := image.NewGray(image.Rect(0, 0, width, height))
	var wg sync.WaitGroup

	rowsPerWorker := height / numWorkers


	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		startRow := i * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if i == numWorkers-1 {
			// Ensure that the last worker processes remaining rows
			endRow = height
		}

		go func(startRow, endRow int) {
			defer wg.Done()
			for y := startRow; y < endRow; y++ {
				for x := 0; x < width; x++ {
					gray.SetGray(x, y, color.Gray{uint8(data[y][x])})
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()

	return gray
}


func combineAndNormalize(gradientX, gradientY [][]float64) [][]float64{

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

	return edges
}


// saveImage saves an image to file
func saveImage(filename string, img image.Image) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Error saving image:", err)
	}
	defer file.Close()

	err = jpeg.Encode(file, img, nil)
	if err != nil {
		log.Fatal("Error encoding image:", err)
	}
}