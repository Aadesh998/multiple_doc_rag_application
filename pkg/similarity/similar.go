package similarity

import (
	"math"
)

type SortableEmbedding struct {
	ID         int
	Chunk      string
	Similarity float64
}

func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64

	for i := 0; i < len(a); i++ {
		av := float64(a[i])
		bv := float64(b[i])

		dotProduct += av * bv
		normA += av * av
		normB += bv * bv
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
