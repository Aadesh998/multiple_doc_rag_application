package handlers

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

var MsgChan = make(chan string, 20)

func WSHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ERROR: WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	defer close(done)

	go func() {
		for {
			select {
			case msg, ok := <-MsgChan:
				if !ok {
					return
				}
				err := conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
				if err != nil {
					log.Printf("ERROR: SetWriteDeadline failed: %v", err)
					return
				}
				if err := conn.WriteMessage(websocket.TextMessage, []byte(msg)); err != nil {
					log.Printf("ERROR: WriteMessage failed: %v", err)
					return
				}
			case <-done:
				return
			case <-ctx.Done():
				return
			}
		}
	}()

	for {
		err := conn.SetReadDeadline(time.Now().Add(2 * time.Hour))
		if err != nil {
			log.Printf("ERROR: SetReadDeadline failed: %v", err)
			break
		}

		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Printf("INFO: WebSocket closed: %v", err)
			break
		}

		var input map[string]interface{}
		if err := json.Unmarshal(msg, &input); err != nil {
			log.Printf("ERROR: Unmarshal failed: %v", err)
			continue
		}

		message, ok := input["message"].(string)
		if !ok {
			log.Printf("WARN: Missing or invalid 'message' field")
			continue
		}

		if message == "stop" {
			log.Printf("INFO: Stop request received")
			cancel()
			ctx, cancel = context.WithCancel(context.Background())
			continue
		}

		MsgChan <- message
	}
}
