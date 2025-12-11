package embedding

import (
	"context"

	"github.com/ollama/ollama/api"
)

func GetEmbeddings(model, prompt string) ([]float64, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	req := &api.EmbeddingRequest{
		Model:  model,
		Prompt: prompt,
	}

	resp, err := client.Embeddings(context.Background(), req)
	if err != nil {
		return nil, err
	}

	return resp.Embedding, nil
}
