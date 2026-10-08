SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
GO ?= go
GOLANGCI_LINT ?= golangci-lint
V ?= 0
export GOWORK := off
export GO
MODULES := $(shell cat scripts/check-modules.txt)
TEST_FLAGS ?=
FUZZ_SECONDS ?= 30
include scripts/toolchain.mk
ifeq ($(V),1)
.SHELLFLAGS := -eux -o pipefail -c
endif

.PHONY: test lint check test-integration examples bench fuzz cover versions fix test-live release-patch release-break prerequisites

test:
	@for module in $(MODULES); do printf '\n[test] %s\n' "$$module"; (cd "$$module" && $(GO) test -race $(TEST_FLAGS) ./...); done

lint:
	@$(GOLANGCI_LINT) config verify
	@for module in $(MODULES); do printf '\n[lint] %s\n' "$$module"; (cd "$$module" && diff=$$($(GOLANGCI_LINT) fmt --diff) || { printf "%s\n" "$$diff"; exit 1; }; test -z "$$diff" || { printf "%s\n" "$$diff"; exit 1; }; $(GOLANGCI_LINT) run --allow-serial-runners ./...); done

prerequisites: versions
	@command -v git >/dev/null
	@command -v docker >/dev/null || { echo 'make check requires Docker' >&2; exit 1; }
	@docker info >/dev/null
	@test -n "$${RAGY_PDF_PYTHON:-}" || { echo 'PDF migration blocked: set RAGY_PDF_PYTHON; see docs/pdf-go-feasibility.md' >&2; exit 1; }
	@command -v "$${RAGY_PDF_PYTHON}" >/dev/null || { echo 'Retained PDF interpreter is unavailable' >&2; exit 1; }
	@cd tooling && $(GO) test -count=1 -run '^TestModuleInventory$$' ./...

check:
	@$(MAKE) --no-print-directory prerequisites
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory test TEST_FLAGS=-count=1
	@$(MAKE) --no-print-directory examples
	@$(MAKE) --no-print-directory test-integration
	@printf '\n[check] PASS\n'

test-integration:
	@printf '\n[integration] PostgreSQL\n'
	@./scripts/postgres-test.sh
	@printf '\n[integration] module artifacts, consumers and release recovery\n'
	@cd tooling && $(GO) test -race -count=1 -timeout=30m -tags=integration ./...
	@$(MAKE) --no-print-directory test-pdf

examples:
	@for module in $(filter examples/%,$(MODULES)); do printf '\n[build] %s\n' "$$module"; (cd "$$module" && $(GO) build ./...); done
	@$(GO) build -o /dev/null ./examples/local-bm25

versions:
	@$(GO) version
	@$(GOLANGCI_LINT) version
	@$(GO) version | grep -Eq ' go$(subst .,[.],$(GO_VERSION)) ' || { echo 'Expected Go $(GO_VERSION)' >&2; exit 1; }
	@$(GOLANGCI_LINT) version | grep -Eq 'version $(subst .,[.],$(LINT_VERSION))( |$$)' || { echo 'Expected golangci-lint $(LINT_VERSION)' >&2; exit 1; }

bench:
	@for module in $(MODULES); do (cd "$$module" && $(GO) test -run='^$$' -bench=. -benchmem ./...); done

fuzz:
	@./scripts/fuzz.sh '$(FUZZ_SECONDS)'

cover:
	@for module in $(MODULES); do (cd "$$module" && $(GO) test -count=1 -coverprofile=coverage.out ./... && $(GO) tool cover -func=coverage.out); done

fix:
	@for module in $(MODULES); do (cd "$$module" && $(GO) fix ./... && $(GOLANGCI_LINT) fmt && $(GOLANGCI_LINT) run --fix ./...); done

test-live:
	@for module in adapters/cohere adapters/openai adapters/gemini adapters/jina; do (cd "$$module" && $(GO) test -race -count=1 -tags=live ./...); done

release-patch:
	@./scripts/release.sh patch '$(RELEASE_SOURCE)'

release-break:
	@./scripts/release.sh break '$(RELEASE_SOURCE)'

.PHONY: test-pdf
test-pdf:
	@printf '\n[integration] PDF\n'
	@test -n "$${RAGY_PDF_PYTHON:-}" || { echo 'PDF migration blocked: set RAGY_PDF_PYTHON for the retained parser; see docs/pdf-go-feasibility.md' >&2; exit 1; }
	@cd adapters/pdf && $(GO) test -race -count=1 -run '^TestActualPDF' ./...
