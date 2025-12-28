ENTRY_POINT=./cmd/main.go

.PHONY: all build 

all: build

build: run 

run:
	@echo "running serer"
	go run ${ENTRY_POINT}