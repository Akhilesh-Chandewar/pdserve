.PHONY: build test race vet fmt ci run repro clean

build:
	go build ./...

test:
	go test ./...

race:
	go test -race ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

# ci is the gate CI runs: format check, vet, race-enabled tests, build.
ci:
	@go vet ./...
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	@go test -race ./...
	@go build ./...

# repro regenerates benchmark/results/sim_results.csv and all charts from
# scratch (M0). Runs are deterministic (Sim clock), so the CSV is stable for
# a given commit and seed set.
repro:
	bash benchmark/sweep.sh
	python3 benchmark/plots.py

run:
	go run ./cmd/pdserve

clean:
	rm -rf bin /tmp/pdserve-bench
