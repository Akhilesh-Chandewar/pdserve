.PHONY: build test vet fmt run clean

build:
	go build ./...

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

ci: vet test

run:
	go run ./cmd/pdserve

clean:
	rm -rf bin
