package web

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ffdc.chat_application/handlers"
)

func StartServer() {
	server := createServer()
	if err := runServer(context.Background(), server, 10*time.Second); err != nil {
		log.Fatalf("Failed to Start Server %s: ", err)
	}
}

func createServer() *http.Server {
	mux := http.DefaultServeMux

	mux.HandleFunc("/api/upload", handlers.ProcessPDFHandler)

	server := &http.Server{
		Addr:    ":5000",
		Handler: mux,
	}

	return server
}

func runServer(
	ctx context.Context,
	server *http.Server,
	shutdownTimeOut time.Duration,
) error {
	errCh := make(chan error, 1)
	go func() {
		log.Println("Server running on :5000")
		if err := server.ListenAndServe(); !errors.Is(
			err, http.ErrServerClosed,
		) {
			errCh <- err
		}
		close(errCh)
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-errCh:
		return err
	case <-stop:
		log.Printf("Received Server Shutdown Signal.")
	case <-ctx.Done():
		log.Printf("Context Time Limit Exceed. Server ShutDown.")
	}

	shutDownCtx, cancel := context.WithTimeout(
		context.Background(),
		shutdownTimeOut,
	)
	defer cancel()

	if err := server.Shutdown(shutDownCtx); err != nil {
		if closeErr := server.Close(); closeErr != nil {
			return errors.Join(err, closeErr)
		}
		return err
	}

	log.Println("Server stopped gracefully")
	return nil
}
