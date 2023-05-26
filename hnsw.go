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
	currentMaxLayer         int
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
		currentMaxLayer:         0,
		ef:                      ef,
		efConstruction:          efConstruction,
		entryPoint:              0, // assuming node ids are always inserted sequentially -- pretty sure this is overwritten anyways
	}
}

/* SEARCH LAYER */
func (h *HNSW) searchLayer(q interface{}, eps []int, ef int, layerNum int) []int {
	visited := make(map[int]struct{})
	// ^ 1- v <-- ep // set of visited elements

	// Initialize candidates and dynamicNearestNeighborsList
	candidates := &distanceHeap{}
	heap.Init(candidates)
	// ^ 2- C <-- ep // set of candidates

	dynamicNearestNeighborsList := &distanceHeap{}
	heap.Init(dynamicNearestNeighborsList)
	// ^ 3 - W <-- ep // dynamic list of found nearest neighbors

	for _, ep := range eps {
		visited[ep] = struct{}{}

		// Maybe this is backwards? -distance candidates, distance W
		heap.Push(candidates, &distanceNode{h.distance(ep, q), ep, candidates.Len()})
		heap.Push(dynamicNearestNeighborsList, &distanceNode{-h.distance(ep, q), ep, dynamicNearestNeighborsList.Len()})
	}
	// ^ 2 & 3, init candidates and foudn nearest neighbors with entry points

	for candidates.Len() > 0 {
		nearest := heap.Pop(candidates).(*distanceNode)
		// ^ 5 c <-- extract nearest element from C to q

		furthest := (*dynamicNearestNeighborsList)[0]
		/* Changed Peeking Code
		furthest := heap.Pop(dynamicNearestNeighborsList).(*distanceNode)
		heap.Push(dynamicNearestNeighborsList, furthest)
		*/
		// ^ 6 f <-- get furthest element from W to q, and put it back in the queue

		if nearest.distance > -furthest.distance {
			break // all elements in dynamicNearestNeighborList are evaluated
		}
		// ^ 7 & 8; if distance(c,q) > distance(f,q) then all elements in W are evaluated

		for _, e := range h.graph[layerNum][nearest.id] { // loop thorugh nearest el's neighborhood
			if _, ok := visited[e]; !ok { // make sure we haven't already visited this node
				visited[e] = struct{}{} // v <-- v union e

				furthest := (*dynamicNearestNeighborsList)[0]
				/* Changed Peeking Code
				furthest = heap.Pop(dynamicNearestNeighborsList).(*distanceNode)
				heap.Push(dynamicNearestNeighborsList, furthest)
				*/

				distanceE := h.distance(e, q)

				if distanceE < -furthest.distance || dynamicNearestNeighborsList.Len() < ef {
					heap.Push(candidates, &distanceNode{distanceE, e, candidates.Len()})
					heap.Push(dynamicNearestNeighborsList, &distanceNode{-distanceE, e, dynamicNearestNeighborsList.Len()})
					if dynamicNearestNeighborsList.Len() > ef {
						heap.Pop(dynamicNearestNeighborsList)
					} // if | W | > ef, then remove furthest element from W to q
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

/* INSERT NODE */
func (h *HNSW) insertNode(qID int, q []float64) {
	// Add the vector to the ID to vector lookup
	h.nodeIDToVector[qID] = q

	/* SET ENTRY POINT WITH MAX LAYER */
	if len(h.graph) == 0 {
		h.graph[0] = make(map[int][]int)
		h.graph[0][qID] = []int{}
		h.entryPoint = qID
		return
	}

	var nearestNeighbors []int
	eps := []int{h.entryPoint}
	L := h.maxLayers
	mL := 1 / math.Log(float64(h.maxConnections))
	l := int(math.Floor(-math.Log(rand.Float64()) * mL))

	/* FIND THE BEST ENTRY POINT FOR A NEW NODE */
	for lc := L; lc > l; lc-- {
		nearestNeighbors = h.searchLayer(q, eps, h.efConstruction, lc)
		// Potential Problem 1 - This could be the problem the searchLayer might not return the nearest neighbor
		eps = nearestNeighbors
		// Weaviate passes in a priority queue to searchLayer, not sure that's a major difference or not
	}

	if l > h.currentMaxLayer {
		for initLayer := h.currentMaxLayer; initLayer <= l; initLayer++ {
			h.graph[initLayer] = make(map[int][]int)
			h.graph[initLayer][qID] = []int{}
		}
		h.currentMaxLayer = l
		h.entryPoint = qID
	}

	// Insert the new element and update connections for layers min(L, 1) down to 0
	for lc := min(L, l); lc >= 0; lc-- {
		nearestNeighbors = h.searchLayer(q, eps, h.efConstruction, lc)
		var neighborsToAdd []int
		if lc == 0 {
			//neighborsToAdd = h.selectNeighborsSimple(qID, nearestNeighbors, h.layerZeroMaxConnections)
			neighborsToAdd = h.selectNeighborsHeuristic(qID, nearestNeighbors, h.layerZeroMaxConnections, lc, true, false)
		} else {
			//neighborsToAdd = h.selectNeighborsSimple(qID, nearestNeighbors, h.maxConnections)
			neighborsToAdd = h.selectNeighborsHeuristic(qID, nearestNeighbors, h.maxConnections, lc, true, false)
		}

		// Add bidirectional connections from neighbors to q at layer lc
		h.addBidirectionalConnections(qID, neighborsToAdd, lc)

		/* Then panics on - `h.graph[lc][neighbor] = []int{}` */

		// Shrink connections if needed
		for _, e := range neighborsToAdd {
			eConn := h.graph[lc][e]
			if len(eConn) > h.maxConnections {
				var eNewConn []int
				if lc == 0 {
					//eNewConn = h.selectNeighborsSimple(e, eConn, h.layerZeroMaxConnections)
					eNewConn = h.selectNeighborsHeuristic(e, eConn, h.layerZeroMaxConnections, lc, true, false)
				} else {
					//eNewConn = h.selectNeighborsSimple(e, eConn, h.maxConnections)
					eNewConn = h.selectNeighborsHeuristic(e, eConn, h.maxConnections, lc, true, false)
				}
				h.graph[lc][e] = eNewConn
			}
		}

		if len(nearestNeighbors) > 0 {
			eps = nearestNeighbors
		}
	}
}

/* CONNECT NEIGHBORS */
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

/* SELECT SIMPLE */
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

/* SELECT HEURISTIC */
func (h *HNSW) selectNeighborsHeuristic(qID int, C []int, M int, lc int, extendCandidates bool, keepPrunedConnections bool) []int {
	R := make([]int, 0) // result set
	// ^ 1- R <-- null
	W := &distanceHeap{} // working queue for the candidates

	// Populate working queue with initial candidates and their distances to query
	for _, e := range C {
		heap.Push(W, &distanceNode{h.distance(qID, e), e, W.Len()})
	}
	// ^ 2 - W <-- C

	// Extend candidates by their neighbors
	// Not sure if this is productive, might be good to A/B test but doubt this is what is causing the recall problem
	if extendCandidates {
		for _, e := range C {
			for _, eAdj := range h.graph[lc][e] {
				// Could I do this with a set to make it faster? -- vs. contains in the heap
				if !W.contains(eAdj) {
					heap.Push(W, &distanceNode{h.distance(qID, eAdj), eAdj, W.Len()})
				}
			}
		}
	}
	// ^ 3 - 7; if extendCandidates
	WDiscarded := &distanceHeap{} // queue for the discarded candidates
	// ^ 8 -- Wd <-- null // queue for the discarded candidates
	for W.Len() > 0 && len(R) < M {
		// Find the point in the working list with the smallest distance
		distE := heap.Pop(W).(*distanceNode)
		// ^ 10 <-- e <- extract nearest element from W to q

		// If the result list R is empty OR
		// If the distance between the query and e is less than
		if len(R) == 0 {
			R = append(R, distE.id)
		} else {
			good := true
			for _, candidate := range R {
				if distE.distance < h.distance(qID, candidate) {
					good = false
					break
				}
			}
			// ^ This is the meat of the pruning heuristic

			if good {
				R = append(R, distE.id) // e is closer to q compared to any el in R
			} else {
				heap.Push(WDiscarded, distE) // if using for keepPrunedConnections...
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

/* SEARCH */
func (h *HNSW) search(q []float64, K int, ef int) []int {
	// Set the initial entry points and layer
	eps := []int{h.entryPoint}
	layer := h.maxLayers

	// Create a list to hold the current nearest elements
	var W []int

	// Traverse from the top layer to layer 1
	for lc := layer; lc > 0; lc-- {
		W = h.searchLayer(q, eps, 1, lc)
		eps = W
	}

	// Search layer 0
	W = h.searchLayer(q, eps, ef, 0)

	// Sort W based on distance to the query and return the K nearest elements
	sort.Slice(W, func(i, j int) bool {
		return h.distance(W[i], q) < h.distance(W[j], q)
	})

	if len(W) > K {
		W = W[:K]
	}

	return W
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

/* DISTANCER */
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

// distanceNode is a node in the distanceHeap
type distanceNode struct {
	distance float64
	id       int
	index    int // add this field to keep track of the index in the heap
}

// distanceHeap is a min-heap of distanceNodes
type distanceHeap []*distanceNode

func (h distanceHeap) Len() int           { return len(h) }
func (h distanceHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h distanceHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *distanceHeap) Push(x interface{}) {
	n := len(*h)
	item := x.(*distanceNode)
	item.index = n
	*h = append(*h, item)
}

func (h *distanceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	item.index = -1 // for safety
	*h = old[0 : n-1]
	return item
}

func (h *distanceHeap) contains(id int) bool {
	for _, node := range *h {
		if node.id == id {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

