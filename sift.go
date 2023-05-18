package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"time"
)

func main() {
	// [...] Previous setup code
	// maxConnections, maxLayers, ef, efConstruction
	hnsw := newHNSW(8, 4, 64, 128)

	//build index
	file, err := os.Open("./sift-data/sift_base.fvecs")
	if err != nil {
		fmt.Println("Error opening file:", err)
		os.Exit(1)
	}
	defer file.Close()

	var dimension int32
	var index_counter int = 0

	// Start timer
	start := time.Now()

	for {
		err = binary.Read(file, binary.LittleEndian, &dimension)
		if err != nil {
			fmt.Println(err)
			break
		}

		vector := make([]float32, dimension)
		err = binary.Read(file, binary.LittleEndian, vector)
		if err != nil {
			fmt.Println(err)
			break
		}

		// Convert vector to []float64
		vector64 := make([]float64, len(vector))
		for i, v := range vector {
			vector64[i] = float64(v)
		}

		hnsw.insertNode(index_counter, vector64)

		index_counter += 1
		if index_counter%10_000 == 9_999 {
			progress_check := time.Since(start)
			fmt.Println(index_counter)
			fmt.Printf("Inserted in %s.\n", progress_check)
		}
	}

	// End timer and print duration
	elapsed := time.Since(start)
	fmt.Printf("Finished building index in %s.\n", elapsed)
	fmt.Println(index_counter)

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

	var queryCounter float64
	var recallSum float64

	for {
		// Read a query vector
		err = binary.Read(queryFile, binary.LittleEndian, &dimension)
		if err != nil {
			break
		}

		queryVector := make([]float32, dimension)
		err = binary.Read(queryFile, binary.LittleEndian, queryVector)
		if err != nil {
			break
		}

		// Convert vector to []float64
		queryVector64 := make([]float64, len(queryVector))
		for i, v := range queryVector {
			queryVector64[i] = float64(v)
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

		// Convert ground truth vector to []int
		groundTruthVectorInt := make([]int, len(groundTruthVector))
		for i, v := range groundTruthVector {
			groundTruthVectorInt[i] = int(v)
		}

		// Search and calculate recall
		searchResults := hnsw.search(queryVector64, 100, 128)
		fmt.Println("Search Results")
		fmt.Println(searchResults)
		fmt.Println("Ground Truth Results")
		fmt.Println(groundTruthVector)
		recall := calculateRecall(searchResults, groundTruthVectorInt)
		recallSum += recall
		queryCounter++
	}

	// Calculate average recal4l
	averageRecall := recallSum / queryCounter
	fmt.Println("Average recall:", averageRecall)
}

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
