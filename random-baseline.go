package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"time"
)

type Vector struct {
	id   int
	data []float64
}

func randomBaseline() {
	rand.Seed(time.Now().UnixNano())

	baseFile := "./sift-data/sift_base.fvecs"

	baseVectors, err := readFvecs(baseFile)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Open the query and ground truth files
	queryFile, err := os.Open("./sift-data/sift_query.fvecs")
	if err != nil {
		fmt.Println("Error opening file:", err)
		os.Exit(1)
	}
	defer queryFile.Close()

	groundTruthFile, err := os.Open("./sift-data/sift_groundtruth.ivecs")
	if err != nil {
		fmt.Println("Error opening file:", err)
		os.Exit(1)
	}
	defer groundTruthFile.Close()

	var recallSum float64
	queryCounter := 0

	for {
		// Read a query vector
		var dimension int32
		err = binary.Read(queryFile, binary.LittleEndian, &dimension)
		if err != nil {
			break
		}

		queryVector := make([]float32, dimension)
		err = binary.Read(queryFile, binary.LittleEndian, queryVector)
		if err != nil {
			break
		}

		// Read a ground truth vector
		err = binary.Read(groundTruthFile, binary.LittleEndian, &dimension)
		if err != nil {
			break
		}

		groundTruthVector := make([]int32, dimension)
		err = binary.Read(groundTruthFile, binary.LittleEndian, groundTruthVector)
		if err != nil {
			break
		}

		// Randomly select a vector from the baseVectors
		result := baseVectors[rand.Intn(len(baseVectors))]

		// Calculate recall
		searchResults := []int{result.id}
		groundTruth := make([]int, len(groundTruthVector))
		for i, v := range groundTruthVector {
			groundTruth[i] = int(v)
		}

		recall := calculateRecall(searchResults, groundTruth)
		recallSum += recall

		queryCounter++
	}

	// Calculate average recall
	averageRecall := recallSum / float64(queryCounter)
	fmt.Printf("Average recall: %f\n", averageRecall)
}

func readFvecs(filepath string) ([]Vector, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var vectors []Vector
	var indexCounter int = 0

	for {
		var dimension int32
		err = binary.Read(file, binary.LittleEndian, &dimension)
		if err != nil {
			break
		}

		vector := make([]float32, dimension)
		err = binary.Read(file, binary.LittleEndian, vector)
		if err != nil {
			break
		}

		// Convert vector to []float64
		vector64 := make([]float64, len(vector))
		for i, v := range vector {
			vector64[i] = float64(v)
		}

		vectors = append(vectors, Vector{id: indexCounter, data: vector64})

		indexCounter++
	}

	return vectors, nil
}

func mai2() {
	randomBaseline()
}
