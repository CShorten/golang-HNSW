package main

import (
	"encoding/binary"
	"fmt"
	"os"
)

func main() {
	hnsw := newHNSW(16, 4, 32, 32)

	//build index
	file, err := os.Open("./sift-data/sift_base.fvecs")
	if err != nil {
		fmt.Println("Error opening file:", err)
		os.Exit(1)
	}
	defer file.Close()

	var dimension int32
	var index_counter int = 0
	var queryVector []float64

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

		if index_counter < 10000 {
			hnsw.insertNode(index_counter, vector64)
		} else if index_counter == 10000 {
			queryVector = vector64
			break
		}

		index_counter += 1
	}

	fmt.Println("Finished building index.")
	fmt.Println(index_counter)

	if len(queryVector) > 0 {
		fmt.Println("Running query...")
		result := hnsw.search(queryVector, 16, 32)
		fmt.Println("Query results:", result)
	} else {
		fmt.Println("No query vector found.")
	}
}
