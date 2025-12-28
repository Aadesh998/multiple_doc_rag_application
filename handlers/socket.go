package handlers

import (
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"

	"ffdc.chat_application/pkg/rag"
	"ffdc.chat_application/pkg/similarity"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type ReRankedChunk struct {
	Chunk      string
	Similarity float64
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

		resp, err := rag.FindSimilarChunksPython(query, 20)
		if err != nil {
			log.Println("Vector search error:", err)
			if err := conn.WriteMessage(websocket.TextMessage, []byte("Error: Vector search failed via Python subprocess. Check app.log for details.")); err != nil {
				log.Println(err)
				return
			}
			continue
		}

		if len(resp.Result) == 0 {
			if err := conn.WriteMessage(websocket.TextMessage, []byte("No relevant information found in the document.")); err != nil {
				log.Println(err)
				return
			}
			continue
		}

		var ranked []ReRankedChunk

		for _, r := range resp.Result {
			score := similarity.CosineSimilarity(resp.QueryVector, r.Embedding)
			ranked = append(ranked, ReRankedChunk{
				Chunk:      r.Chunk,
				Similarity: score,
			})
		}

		sort.Slice(ranked, func(i, j int) bool {
			return ranked[i].Similarity > ranked[j].Similarity
		})

		topK := 5
		if len(ranked) < topK {
			topK = len(ranked)
		}

		var contextParts []string
		for i := 0; i < topK; i++ {
			contextParts = append(contextParts, ranked[i].Chunk)
		}

		for _, similarChunk := range contextParts {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(similarChunk)); err != nil {
				log.Println(err)
				return
			}
		}

		context := strings.Join(contextParts, "\n\n")

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
