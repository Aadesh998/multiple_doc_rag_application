package similarity

import (
	"math"
)

type SortableEmbedding struct {
	ID         int
	Chunk      string
	Similarity float64
}

func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func MergeSort(items []SortableEmbedding) []SortableEmbedding {
	if len(items) < 2 {
		return items
	}

	mid := len(items) / 2
	left := MergeSort(items[:mid])
	right := MergeSort(items[mid:])

	return merge(left, right)
}

func merge(left, right []SortableEmbedding) []SortableEmbedding {
	var result []SortableEmbedding
	i, j := 0, 0

	for i < len(left) && j < len(right) {
		if left[i].Similarity >= right[j].Similarity {
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}

	result = append(result, left[i:]...)
	result = append(result, right[j:]...)

	return result
}
