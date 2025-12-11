package database

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"log"
	"math"

	_ "modernc.org/sqlite"
)

var DB *sql.DB

type Embedding struct {
	ID        int
	Chunk     string
	Embedding []float64
}

func InitDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, err
	}
	DB = db
	return db, nil
}

func IsDBPopulated(db *sql.DB) (bool, error) {
	var count int
	err := db.QueryRow("SELECT COUNT(*) FROM embeddings").Scan(&count)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

func LoadEmbeddings(db *sql.DB) ([]Embedding, error) {
	rows, err := db.Query("SELECT id, chunk, embedding FROM embeddings")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var embeddings []Embedding

	for rows.Next() {
		var emb Embedding
		var blob []byte

		if err := rows.Scan(&emb.ID, &emb.Chunk, &blob); err != nil {
			log.Printf("Error scanning row: %v", err)
			continue
		}

		floats, err := BytesToFloat32Slice(blob)
		if err != nil {
			log.Printf("Error decoding embedding: %v", err)
			continue
		}

		emb.Embedding = floats
		embeddings = append(embeddings, emb)
	}

	return embeddings, nil
}

func BytesToFloat32Slice(b []byte) ([]float64, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("invalid byte length for float32 array")
	}

	count := len(b) / 4
	floats := make([]float64, count)

	for i := 0; i < count; i++ {
		bits := binary.LittleEndian.Uint32(b[i*4 : (i+1)*4])
		floats[i] = float64(math.Float32frombits(bits))
	}

	return floats, nil
}
