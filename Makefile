SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
GO ?= go
GOLANGCI_LINT ?= golangci-lint
.DEFAULT_GOAL := test
V ?= 0
export GOWORK := off
export GO
MODULES := $(sort $(patsubst ./%,%,$(shell find . -type d \( -name '.*' ! -name '.' -o -name vendor \) -prune -o -type f -name go.mod -exec dirname {} \;)))
TEST_FLAGS ?=
FUZZ_SECONDS ?= 30
include scripts/toolchain.mk
ifeq ($(V),1)
.SHELLFLAGS := -eux -o pipefail -c
endif

.PHONY: test lint check test-integration examples bench fuzz cover versions fix test-live modules release-patch release-break prerequisites

test:
	@for module in $(MODULES); do \
		printf '\n[test] %s\n' "$$module"; \
		(cd "$$module" && $(GO) test -race $(TEST_FLAGS) ./...); \
	done

modules:
	@printf '%s\n' $(MODULES)

lint:
	@$(GOLANGCI_LINT) config verify
	@for module in $(MODULES); do \
		printf '\n[lint] %s\n' "$$module"; \
		(cd "$$module" && $(GOLANGCI_LINT) fmt --diff && $(GOLANGCI_LINT) run --allow-serial-runners ./...); \
	done

prerequisites: versions
	@command -v git >/dev/null
	@$(MAKE) --no-print-directory prerequisites-project

check:
	@$(MAKE) --no-print-directory prerequisites
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory test TEST_FLAGS=-count=1
	@$(MAKE) --no-print-directory examples
	@$(MAKE) --no-print-directory test-integration
	@printf '\n[check] PASS\n'

test-integration:
	@$(MAKE) --no-print-directory check-project

examples:
	@$(MAKE) --no-print-directory examples-project

versions:
	@version=$$($(GO) version); printf '%s\n' "$$version"; \
		[[ "$$version" == 'go version go$(GO_VERSION) '* ]] || { echo 'Expected Go $(GO_VERSION)' >&2; exit 1; }
	@version=$$($(GOLANGCI_LINT) version); printf '%s\n' "$$version"; \
		[[ "$$version" == *'version $(LINT_VERSION) '* ]] || { echo 'Expected golangci-lint $(LINT_VERSION)' >&2; exit 1; }

bench:
	@for module in $(MODULES); do \
		printf '\n[bench] %s\n' "$$module"; \
		(cd "$$module" && $(GO) test -run='^$$' -bench=. -benchmem ./...); \
	done

fuzz:
	@./scripts/fuzz.sh '$(FUZZ_SECONDS)' $(MODULES)

cover:
	@for module in $(MODULES); do \
		printf '\n[cover] %s\n' "$$module"; \
		(cd "$$module" && $(GO) test -count=1 -coverprofile=coverage.out ./... && $(GO) tool cover -func=coverage.out); \
	done

fix:
	@for module in $(MODULES); do \
		printf '\n[fix] %s\n' "$$module"; \
		(cd "$$module" && $(GO) fix ./... && $(GOLANGCI_LINT) fmt && $(GOLANGCI_LINT) run --fix ./...); \
	done

test-live:
	@for module in $(MODULES); do \
		printf '\n[live] %s\n' "$$module"; \
		(cd "$$module" && $(GO) test -race -count=1 -tags=live -run='^TestLive' ./...); \
	done

release-patch:
	@./scripts/release.sh patch '$(RELEASE_SOURCE)'

release-break:
	@./scripts/release.sh break '$(RELEASE_SOURCE)'

release-inspect release-resume release-finish:
	@./scripts/release.sh $(patsubst release-%,%,$@)

.PHONY: prerequisites-project examples-project check-project release-inspect release-resume release-finish
prerequisites-project examples-project check-project:

-include project.mk
