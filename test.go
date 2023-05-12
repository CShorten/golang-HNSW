package main

import (
	"fmt"
)

func main() {
	testHNSW()
}

func testHNSW() {
	hnsw := newHNSW(16, 4, 50, 200)

	data := [][]float64{
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 3.0},
		{4.0, 4.0},
		{5.0, 5.0},
		{6.0, 6.0},
	}

	for i, point := range data {
		hnsw.insertNode(i, point)
	}

	query := []float64{3.5, 3.5}
	k := 1
	ef := 50

	fmt.Println("HNSW Graph:")
	hnsw.prettyPrintGraph()

	result := hnsw.search(query, k, ef)
	fmt.Printf("Searching for the nearest neighbor of the query point %v\n", query)
	fmt.Printf("The nearest neighbor ID is: %d\n", result[0])
	fmt.Printf("The nearest neighbor coordinates are: %v\n", hnsw.nodeIDToVector[result[0]])
}
