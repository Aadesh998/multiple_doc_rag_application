package rag

import (
	"context"
	"fmt"

	"ffdc.chat_application/pkg/database"
	"ffdc.chat_application/pkg/similarity"
	"github.com/ollama/ollama/api"
)

func FindSimilarChunks(queryEmbedding []float64, vectorStore []database.Embedding, topK int) []string {
	if len(vectorStore) == 0 {
		return nil
	}

	var similarChunks []similarity.SortableEmbedding
	for _, emb := range vectorStore {
		sim := similarity.CosineSimilarity(queryEmbedding, emb.Embedding)
		similarChunks = append(similarChunks, similarity.SortableEmbedding{
			ID:         emb.ID,
			Chunk:      emb.Chunk,
			Similarity: sim,
		})
	}

	sortedChunks := similarity.MergeSort(similarChunks)

	var topChunks []string
	for i := 0; i < topK && i < len(sortedChunks); i++ {
		topChunks = append(topChunks, sortedChunks[i].Chunk)
	}

	return topChunks
}

func GenerateResponse(chatModel, prompt string) (string, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return "", err
	}
	value := true
	req := &api.ChatRequest{
		Model:  chatModel,
		Stream: &value,
		Messages: []api.Message{
			{Role: "user", Content: prompt},
		},
		Options: map[string]interface{}{
			"num_predict": 1024,
		},
	}

	ctx := context.Background()
	var responseContent string
	err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
		fmt.Print(resp.Message.Content)
		responseContent += resp.Message.Content
		return nil
	})
	if err != nil {
		return "", err
	}

	fmt.Println()
	return responseContent, nil
}
