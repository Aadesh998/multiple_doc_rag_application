package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"

	"github.com/ollama/ollama/api"
)

func FindSimilarChunksPython(query string, topK int) ([]string, error) {
	cmd := exec.Command("python", "processPDF.py", "--search-go", query, fmt.Sprintf("%d", topK))
	output, err := cmd.Output()
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			log.Printf("Python script failed (vector search). Stderr: %s", string(exitError.Stderr))
		}
		return nil, fmt.Errorf("error executing python search: %w", err)
	}

	var chunks []string
	if err := json.Unmarshal(output, &chunks); err != nil {
		log.Printf("Failed to decode Python output (raw output: %s)", string(output))
		return nil, fmt.Errorf("error decoding python search output: %w", err)
	}

	return chunks, nil
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
