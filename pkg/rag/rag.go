package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"

	"github.com/ollama/ollama/api"
)

type SimilarChunk struct {
	Chunk     string    `json:"chunk"`
	Embedding []float32 `json:"embedding"`
}

type SearchResponse struct {
	QueryVector []float32      `json:"query_vector"`
	Result      []SimilarChunk `json:"result"`
}

func FindSimilarChunksPython(query string, topK int) (SearchResponse, error) {
	cmd := exec.Command("python", "processPDF.py", "--search-go", query, fmt.Sprintf("%d", topK))

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		log.Printf("Python stderr: %s", stderr.String())
		return SearchResponse{}, fmt.Errorf("python execution failed: %w", err)
	}

	var response SearchResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		log.Printf("Failed to decode Python output (raw output: %s)", string(err.Error()))
		return SearchResponse{}, fmt.Errorf("error decoding python search output: %w", err)
	}

	return response, nil
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
		Options: map[string]any{
			"num_predict": 2048,
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
