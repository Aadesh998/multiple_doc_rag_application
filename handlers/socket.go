package handlers

import (
	"fmt"
	"log"
	"net/http"
	"strings"

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

func HandleWebSocket(w http.ResponseWriter, r *http.Request) {
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

		similarChunks, err := rag.FindSimilarChunksPython(query, 5)
		if err != nil {
			log.Println("Vector search error:", err)
			if err := conn.WriteMessage(websocket.TextMessage, []byte("Error: Vector search failed via Python subprocess. Check app.log for details.")); err != nil {
				log.Println(err)
				return
			}
			continue
		}

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
		prompt := fmt.Sprintf(`You are given the following retrieved context:
        %s

        Based on this context, answer the user’s query:
        %s

        Guidelines:
        - Use only the information from the context to answer.
        - If the context includes multiple points, present them clearly in separate lines or bullet points use : after title and \n if the line ends.
        - If the answer is not directly in the context, say so rather than making it up.
        - Keep the response concise, clear, and directly relevant to the query.`, context, query)

		response, err := rag.GenerateResponse("mistral:7b-instruct-q4_K_M", prompt)
		if err != nil {
			log.Printf("Error generating response from chat model: %v", err)
			if err := conn.WriteMessage(websocket.TextMessage, []byte("Error generating response.")); err != nil {
				log.Println(err)
				return
			}
			continue
		}

		if err := conn.WriteMessage(websocket.TextMessage, []byte("======================")); err != nil {
			log.Println(err)
			return
		}

		if err := conn.WriteMessage(websocket.TextMessage, []byte(response)); err != nil {
			log.Println(err)
			return
		}
	}
}
