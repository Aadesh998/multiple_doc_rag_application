package handlers

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
)

type PDFResponse struct {
	Status  string   `json:"status"`
	Message string   `json:"message"`
	Files   []string `json:"files"`
}

func ProcessPDFHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(20 << 20)
	if err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	files := r.MultipartForm.File["file"]
	if len(files) == 0 {
		http.Error(w, "no files uploaded", http.StatusBadRequest)
		return
	}

	saveDir := "./uploads"
	_ = os.MkdirAll(saveDir, os.ModePerm)

	var (
		savedFiles []string
		output     []byte
	)

	for _, fileHeader := range files {
		file, err := fileHeader.Open()
		if err != nil {
			http.Error(w, "failed to open file", http.StatusInternalServerError)
			return
		}
		defer file.Close()

		dstPath := filepath.Join(saveDir, filepath.Base(fileHeader.Filename))
		dst, err := os.Create(dstPath)
		if err != nil {
			http.Error(w, "failed to create file", http.StatusInternalServerError)
			return
		}

		_, err = io.Copy(dst, file)
		dst.Close()
		if err != nil {
			http.Error(w, "failed to save file", http.StatusInternalServerError)
			return
		}

		savedFiles = append(savedFiles, dstPath)

		cmd := exec.Command("python3", "process_pdf.py", dstPath)
		output, err = cmd.CombinedOutput()
		if err != nil {
			log.Println("Python error:", string(output))
			http.Error(w, string(output), http.StatusInternalServerError)
			return
		}
	}

	resp := PDFResponse{
		Status:  "success",
		Message: "PDF processed & embeddings saved",
		Files:   savedFiles,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
