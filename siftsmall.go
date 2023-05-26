package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"time"
)

// Function to read SIFT formatted vector data from a given path
func ReadSiftVecsFrom(path string, size int, dimensions int) [][]float32 {
	// print progress
	fmt.Printf("generating %d vectors...", size)

	// read the vectors
	vectors := readSiftFloat(path, size, dimensions)

	// print completion
	fmt.Printf(" done\n")

	// return the vectors
	return vectors
}

// Function to read base and query vector data from a given path
func ReadVecs(size int, queriesSize int, dimensions int, db string, path ...string) ([][]float32, [][]float32) {
	// print progress
	fmt.Printf("generating %d vectors...", size+queriesSize)

	// set the base uri as db
	uri := db

	// if a path is provided, prepend it to uri
	if len(path) > 0 {
		uri = fmt.Sprintf("%s/%s", path[0], uri)
	}

	// read base vectors
	vectors := readSiftFloat(fmt.Sprintf("siftsmall/%s_base.fvecs", db), size, dimensions)

	// read query vectors
	queries := readSiftFloat(fmt.Sprintf("siftsmall/%s_query.fvecs", db), queriesSize, dimensions)

	// print completion
	fmt.Printf(" done\n")

	// return vectors and queries
	return vectors, queries
}

func ReadSiftIVecsFrom(filename string, vectorLengthInt int32) ([][]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	vectors := [][]int{}

	for {
		vector, err := readSiftInt(file, vectorLengthInt)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vector)
	}

	return vectors, nil
}

func readSiftInt(r io.Reader, vectorLengthInt int32) ([]int, error) {
	// read vector dimension
	var dimension int32
	if err := binary.Read(r, binary.LittleEndian, &dimension); err != nil {
		return nil, err
	}
	if dimension != vectorLengthInt {
		return nil, fmt.Errorf("Each vector must have %d entries.", vectorLengthInt)
	}

	// read vector
	vector := make([]int32, dimension)
	if err := binary.Read(r, binary.LittleEndian, vector); err != nil {
		return nil, err
	}

	// convert vector to []int
	vectorInt := make([]int, len(vector))
	for i, v := range vector {
		vectorInt[i] = int(v)
	}
	return vectorInt, nil
}

// Function to read SIFT formatted vector data from a given binary file
func readSiftFloat(file string, maxObjects int, vectorLengthFloat int) [][]float32 {
	// open the file
	f, err := os.Open(file)

	// ensure file gets closed after the function exits
	defer f.Close()

	// check for file open error
	if err != nil {
		panic(err)
	}

	// Allocate memory for objects and vectorBytes
	objects := make([][]float32, maxObjects)
	vectorBytes := make([]byte, 4+vectorLengthFloat*4)

	// read the vectors from the file
	for i := 0; i >= 0; i++ {
		_, err = f.Read(vectorBytes)

		// break the loop if we have reached end of file
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}

		// check if the vector length matches expected length
		if int32FromBytes(vectorBytes[0:4]) != vectorLengthFloat {
			panic("Each vector must have 128 entries.")
		}

		// read each float from the vector
		vectorFloat := make([]float32, vectorLengthFloat)
		for j := 0; j < vectorLengthFloat; j++ {
			start := (j + 1) * 4 // first 4 bytes are length of vector
			vectorFloat[j] = float32FromBytes(vectorBytes[start : start+4])
		}

		// save the vector
		objects[i] = vectorFloat

		// break the loop if we have reached maximum number of objects
		if i >= maxObjects-1 {
			break
		}
	}

	// return the objects read from file
	return objects
}

// Function to convert a byte slice to int
func int32FromBytes(bytes []byte) int {
	return int(binary.LittleEndian.Uint32(bytes))
}

// Function to convert a byte slice to float32
func float32FromBytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

// Main function of the program
func main() {
	// Create a new HNSW instance with certain parameters

	// maxConnections, maxLayers, ef, efConstruction
	hnsw := newHNSW(16, 4, 128, 256) // 32 resulted in 99.7

	// Read base vectors from file
	vectors := ReadSiftVecsFrom("./siftsmall/siftsmall_base.fvecs", 10000, 128)
	index_counter := 0

	// Start timer for performance measurement
	start := time.Now()

	// Add each vector to the HNSW index
	for _, vector := range vectors {
		// Convert float32 slice to float64
		vector64 := float32SliceToFloat64(vector)

		// Insert vector into hnsw
		hnsw.insertNode(index_counter, vector64)

		index_counter++
		if index_counter%10_000 == 9_999 {
			progress_check := time.Since(start)
			fmt.Println(index_counter)
			fmt.Printf("Inserted in %s.\n", progress_check)
		}
	}

	// Print duration for building index
	elapsed := time.Since(start)
	fmt.Printf("Finished building index in %s.\n", elapsed)
	fmt.Println(index_counter)

	// Read the query vectors and ground truth vectors from files
	_, queryVectors := ReadVecs(10000, 100, 128, "siftsmall")
	groundTruthVectors, _ := ReadSiftIVecsFrom("./siftsmall/siftsmall_groundtruth.ivecs", 100)

	var queryCounter float64
	var recallSum float64

	// For each query vector, perform search and calculate recall
	for i := range queryVectors {
		// Convert float32 slice to float64
		queryVector64 := float32SliceToFloat64(queryVectors[i])

		// Convert float32 slice to int
		groundTruthVector := groundTruthVectors[i]

		// Perform search
		searchResults := hnsw.search(queryVector64, 100, 512)
		/*
			fmt.Println("HNSW Results")
			fmt.Println(searchResults)
			fmt.Println("Ground Truth")
			fmt.Println(groundTruthVector)
		*/

		// Calculate recall
		recall := calculateRecall(searchResults, groundTruthVector)
		recallSum += recall
		queryCounter++
	}

	// Calculate and print average recall
	averageRecall := recallSum / queryCounter
	fmt.Println("Average recall:", averageRecall)
}

// Function to convert a float32 slice to a float64 slice
func float32SliceToFloat64(float32Slice []float32) []float64 {
	float64Slice := make([]float64, len(float32Slice))
	for i, v := range float32Slice {
		float64Slice[i] = float64(v)
	}
	return float64Slice
}

// Function to convert a float32 slice to an int slice
func int32SliceToInt(int32Slice []float32) []int {
	intSlice := make([]int, len(int32Slice))
	for i, v := range int32Slice {
		intSlice[i] = int(v)
	}
	return intSlice
}

// Function to calculate recall
func calculateRecall(searchResults []int, groundTruth []int) float64 {
	count := 0
	for _, val := range searchResults {
		for _, truth := range groundTruth {
			if val == truth {
				count++
				break
			}
		}
	}
	return float64(count) / float64(len(groundTruth))
}
