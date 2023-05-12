package main

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

type HNSW struct {
	nodeIDToVector          map[int][]float64
	graph                   map[int]map[int][]int
	maxConnections          int
	layerZeroMaxConnections int
	maxLayers               int
	ef                      int
	efConstruction          int
	entryPoint              int
}

func newHNSW(maxConnections int, maxLayers int, ef int, efConstruction int) *HNSW {
	return &HNSW{
		nodeIDToVector:          make(map[int][]float64),
		graph:                   make(map[int]map[int][]int),
		maxConnections:          maxConnections,
		layerZeroMaxConnections: maxConnections * 2,
		maxLayers:               maxLayers,
		ef:                      ef,
		efConstruction:          efConstruction,
		entryPoint:              0,
	}
}

// Distance calculates the Euclidean distance between two vectors
func (h *HNSW) distance(a interface{}, b interface{}) float64 {
	var aVector, bVector []float64

	// Check if a is an int or a slice
	switch v := a.(type) {
	case int:
		aVector = h.nodeIDToVector[v]
	case []float64:
		aVector = v
	default:
		panic("a must be either an int or a []float64")
	}

	// Check if b is an int or a slice
	switch v := b.(type) {
	case int:
		bVector = h.nodeIDToVector[v]
	case []float64:
		bVector = v
	default:
		panic("b must be either an int or a []float64")
	}

	return euclideanDistance(aVector, bVector)
}

// EuclideanDistance calculates the Euclidean distance between two vectors
func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		delta := a[i] - b[i]
		sum += delta * delta
	}
	return math.Sqrt(sum)
}

// SearchLayer searches the graph on a single layer for the nearest neighbors of q
func (h *HNSW) searchLayer(q interface{}, ep int, ef int, layerNum int) []int {
	visited := make(map[int]struct{})
	visited[ep] = struct{}{}

	// Initialize candidates and dynamicNearestNeighborsList
	candidates := &distanceHeap{}
	heap.Init(candidates)
	heap.Push(candidates, &distanceNode{-h.distance(ep, q), ep})

	dynamicNearestNeighborsList := &distanceHeap{}
	heap.Init(dynamicNearestNeighborsList)
	heap.Push(dynamicNearestNeighborsList, &distanceNode{-math.Inf(1), ep})

	for candidates.Len() > 0 {
		nearest := heap.Pop(candidates).(*distanceNode)

		furthest := heap.Pop(dynamicNearestNeighborsList).(*distanceNode)
		heap.Push(dynamicNearestNeighborsList, furthest)

		if -nearest.distance > -furthest.distance {
			break
		}

		for _, e := range h.graph[layerNum][nearest.id] {
			if _, ok := visited[e]; !ok {
				visited[e] = struct{}{}
				distanceE := h.distance(e, q)

				furthest = heap.Pop(dynamicNearestNeighborsList).(*distanceNode)
				heap.Push(dynamicNearestNeighborsList, furthest)

				if distanceE < -furthest.distance || dynamicNearestNeighborsList.Len() < ef {
					heap.Push(candidates, &distanceNode{-distanceE, e})
					heap.Push(dynamicNearestNeighborsList, &distanceNode{-distanceE, e})
					if dynamicNearestNeighborsList.Len() > ef {
						heap.Pop(dynamicNearestNeighborsList)
					}
				}
			}
		}
	}

	result := make([]int, dynamicNearestNeighborsList.Len())
	for i := range result {
		result[i] = heap.Pop(dynamicNearestNeighborsList).(*distanceNode).id
	}

	// Reverse the result because heap returns them in ascending order but we want descending
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}

	return result
}

func (h *HNSW) insertNode(qID int, q []float64) {
	// Add the node to the graph
	h.nodeIDToVector[qID] = q

	// If the graph is empty, insert the first node into each layer and set the entry point
	if len(h.graph) == 0 {
		for initLayer := 0; initLayer <= h.maxLayers; initLayer++ {
			h.graph[initLayer] = make(map[int][]int)
			h.graph[initLayer][qID] = []int{}
		}
		h.entryPoint = qID
		return
	}

	var nearestNeighbors []int
	ep := h.entryPoint
	L := h.maxLayers
	mL := 1 / math.Log(float64(h.maxConnections))
	l := int(math.Floor(-math.Log(rand.Float64()) * mL))

	// Find the entry point from the max layer down to l+1
	for lc := L; lc > l; lc-- {
		nearestNeighbors = h.searchLayer(q, ep, h.efConstruction, lc)
		ep = nearestNeighbors[0]
	}

	// Insert the new element and update connections for layers min(L, 1) down to 0
	for lc := min(L, l); lc >= 0; lc-- {
		nearestNeighbors = h.searchLayer(q, ep, h.efConstruction, lc)
		var neighborsToAdd []int
		if lc == 0 {
			neighborsToAdd = h.selectNeighborsHeuristic(qID, nearestNeighbors, h.layerZeroMaxConnections, lc, false, false)
		} else {
			neighborsToAdd = h.selectNeighborsHeuristic(qID, nearestNeighbors, h.maxConnections, lc, false, false)
		}

		// Add bidirectional connections from neighbors to q at layer lc
		h.addBidirectionalConnections(qID, neighborsToAdd, lc)

		// Shrink connections if needed
		for _, e := range neighborsToAdd {
			eConn := h.graph[lc][e]
			if len(eConn) > h.maxConnections {
				var eNewConn []int
				if lc == 0 {
					eNewConn = h.selectNeighborsHeuristic(e, eConn, h.layerZeroMaxConnections, lc, false, false)
				} else {
					eNewConn = h.selectNeighborsHeuristic(e, eConn, h.maxConnections, lc, false, false)
				}
				h.graph[lc][e] = eNewConn
			}
		}

		if len(nearestNeighbors) > 0 {
			ep = nearestNeighbors[0]
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (h *HNSW) addBidirectionalConnections(qID int, neighborsToAdd []int, lc int) {
	// Add q as a neighbor to each of the nodes in neighborsToAdd at level lc
	for _, neighbor := range neighborsToAdd {
		if _, exists := h.graph[lc][neighbor]; !exists {
			h.graph[lc][neighbor] = []int{}
		}
		h.graph[lc][neighbor] = append(h.graph[lc][neighbor], qID)
	}

	// Add each neighbor in neighborsToAdd to q's connections at level lc
	if _, exists := h.graph[lc][qID]; !exists {
		h.graph[lc][qID] = []int{}
	}
	h.graph[lc][qID] = append(h.graph[lc][qID], neighborsToAdd...)
}

func (h *HNSW) selectNeighborsSimple(qID int, C []int, M int) []int {
	// Compute the distance between q and each candidate element in C
	distances := make([]struct {
		distance  float64
		candidate int
	}, len(C))

	for i, candidate := range C {
		distances[i].distance = h.distance(h.nodeIDToVector[qID], h.nodeIDToVector[candidate])
		distances[i].candidate = candidate
	}

	// Sort the candidate elements by distance to q (ascending order)
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// Select the M nearest elements to q
	M_nearest_elements := make([]int, M)
	for i := 0; i < M && i < len(distances); i++ {
		M_nearest_elements[i] = distances[i].candidate
	}

	return M_nearest_elements
}

func (h *HNSW) selectNeighborsHeuristic(qID int, C []int, M int, lc int, extendCandidates bool, keepPrunedConnections bool) []int {
	R := make([]int, 0)  // result set
	W := &distanceHeap{} // working queue for the candidates

	// Populate working queue with initial candidates and their distances to query
	for _, e := range C {
		heap.Push(W, &distanceNode{h.distance(qID, e), e})
	}

	// Extend candidates by their neighbors
	if extendCandidates {
		for _, e := range C {
			for _, eAdj := range h.graph[lc][e] {
				if !W.contains(eAdj) {
					heap.Push(W, &distanceNode{h.distance(qID, eAdj), eAdj})
				}
			}
		}
	}

	WDiscarded := &distanceHeap{} // queue for the discarded candidates

	for W.Len() > 0 && len(R) < M {
		// Find the point in the working list with the smallest distance
		distE := heap.Pop(W).(*distanceNode)

		// If the result list R is empty OR
		// If the distance between the query and e is less than
		if len(R) == 0 {
			R = append(R, distE.id)
		} else {
			flag := true
			for _, candidate := range R {
				if distE.distance < h.distance(qID, candidate) {
					flag = false
					break
				}
			}

			if flag {
				R = append(R, distE.id)
			} else {
				heap.Push(WDiscarded, distE)
			}
		}
	}

	// Add some of the discarded connections from WDiscarded
	if keepPrunedConnections {
		for WDiscarded.Len() > 0 && len(R) < M {
			distE := heap.Pop(WDiscarded).(*distanceNode)
			R = append(R, distE.id)
		}
	}

	return R
}

func (h *distanceHeap) contains(id int) bool {
	for _, node := range *h {
		if node.id == id {
			return true
		}
	}
	return false
}

func (h *HNSW) prettyPrintGraph() {
	for layer, layerGraph := range h.graph {
		fmt.Printf("Layer %d:\n", layer)
		for nodeID, neighbors := range layerGraph {
			fmt.Printf("Node %d: ", nodeID)
			for _, neighbor := range neighbors {
				fmt.Printf("%d ", neighbor)
			}
			fmt.Println()
		}
		fmt.Println()
	}
}

func (h *HNSW) search(q []float64, K int, ef int) []int {
	// Set the initial entry point and layer
	ep := h.entryPoint
	layer := h.maxLayers

	// Create a list to hold the current nearest elements
	var W []int

	// Traverse from the top layer to layer 1
	for lc := layer; lc > 0; lc-- {
		W = h.searchLayer(q, ep, 1, lc)
		ep = W[0]
		for _, id := range W {
			if h.distance(id, q) < h.distance(ep, q) {
				ep = id
			}
		}
	}

	// Search layer 0
	W = h.searchLayer(q, ep, ef, 0)

	// Sort W based on distance to the query and return the K nearest elements
	sort.Slice(W, func(i, j int) bool {
		return h.distance(W[i], q) < h.distance(W[j], q)
	})

	if len(W) > K {
		W = W[:K]
	}

	return W
}

// distanceNode is a node in the distanceHeap
type distanceNode struct {
	distance float64
	id       int
}

// distanceHeap is a min-heap of distanceNodes
type distanceHeap []*distanceNode

func (h distanceHeap) Len() int           { return len(h) }
func (h distanceHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h distanceHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *distanceHeap) Push(x interface{}) {
	item := x.(*distanceNode)
	*h = append(*h, item)
}

func (h *distanceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // avoid memory leak
	*h = old[0 : n-1]
	return item
}
