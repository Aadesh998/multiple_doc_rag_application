package handlers

import (
	"fmt"
	"log"
	"net/http"
	"strings"

	"ffdc.chat_application/pkg/database"
	"ffdc.chat_application/pkg/embedding"
	"ffdc.chat_application/pkg/rag"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func HandleWebSocket(vectorStore []database.Embedding) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println(err)
			return
		}
		defer conn.Close()

		log.Println("Client connected")

		for {
			_, p, err := conn.ReadMessage()
			if err != nil {
				log.Println(err)
				return
			}

			query := string(p)
			queryEmbedding, err := embedding.GetEmbeddings("nomic-embed-text", query)
			if err != nil {
				log.Printf("Error generating query embedding: %v", err)
				continue
			}

			similarChunks := rag.FindSimilarChunks(queryEmbedding, vectorStore, 3)
			if len(similarChunks) == 0 {
				if err := conn.WriteMessage(websocket.TextMessage, []byte("No relevant information found in the document.")); err != nil {
					log.Println(err)
					return
				}
				continue
			} else {
				for i := 0; i < len(similarChunks); i++ {
					if err := conn.WriteMessage(websocket.TextMessage, []byte(similarChunks[i])); err != nil {
						log.Println(err)
						return
					}
				}
			}

			context := strings.Join(similarChunks, "\n\n")
			prompt := fmt.Sprintf("Based on the following context, answer the user's query.\n\nContext:\n%s\n\nQuery: %s", context, query)

			response, err := rag.GenerateResponse("mistral:7b-instruct-q4_K_M", prompt)
			if err != nil {
				log.Printf("Error generating response from chat model: %v", err)
				if err := conn.WriteMessage(websocket.TextMessage, []byte("Error generating response.")); err != nil {
					log.Println(err)
					return
				}
				continue
			}

			if err := conn.WriteMessage(websocket.TextMessage, []byte(response)); err != nil {
				log.Println(err)
				return
			}
		}
	}
}
