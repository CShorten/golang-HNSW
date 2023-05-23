package main

import (
	"fmt"
	"math/rand"
	"sort"
)

func main() {
	// Number of points
	N := 200

	// Create an HNSW
	h := newHNSW(16, 4, 32, 128)

	// Random seed for reproducibility
	rand.Seed(42)

	// Generate N 2D points and insert them into the graph
	for i := 0; i < N; i++ {
		point := []float64{rand.Float64() * 100, rand.Float64() * 100}
		h.insertNode(i, point)
	}

	// Search point
	q := []float64{50, 50}

	// Perform a search for the 10 nearest neighbors
	hnswNeighbors := h.search(q, 10, 64)
	fmt.Println("Neighbors from HNSW:", hnswNeighbors)

	// Calculate the actual 10 nearest neighbors
	type neighbor struct {
		id       int
		distance float64
	}
	neighborsBrute := make([]neighbor, N)
	for i := 0; i < N; i++ {
		neighborsBrute[i] = neighbor{i, euclideanDistance(q, h.nodeIDToVector[i])}
	}
	sort.Slice(neighborsBrute, func(i, j int) bool {
		return neighborsBrute[i].distance < neighborsBrute[j].distance
	})
	actualNeighbors := neighborsBrute[:10]

	// Convert to []int with ids
	actualNeighborIds := make([]int, len(actualNeighbors))
	for i, neighbor := range actualNeighbors {
		actualNeighborIds[i] = neighbor.id
	}
	fmt.Println("Actual Neighbors:", actualNeighborIds)

	numCorrect := calculateCorrectNeighbors(hnswNeighbors, actualNeighborIds)

	fmt.Printf("Number of correct neighbors found: %d\n", numCorrect)

}

func calculateCorrectNeighbors(hnswNeighbors []int, actualNeighbors []int) int {
	hnswSet := make(map[int]bool)
	actualSet := make(map[int]bool)

	for _, neighbor := range hnswNeighbors {
		hnswSet[neighbor] = true
	}

	for _, neighbor := range actualNeighbors {
		actualSet[neighbor] = true
	}

	var correctNeighbors []int
	for neighbor := range hnswSet {
		if actualSet[neighbor] {
			correctNeighbors = append(correctNeighbors, neighbor)
		}
	}

	return len(correctNeighbors)
}
