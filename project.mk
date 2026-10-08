# ragy-specific checks. The common Makefile also works without this file.
.PHONY: prerequisites-project examples-project check-project test-pdf release-prepare-project release-check-project release-published-project

prerequisites-project:
	@command -v docker >/dev/null || { echo 'make check requires Docker' >&2; exit 1; }
	@docker info >/dev/null
	@test -n "$${RAGY_PDF_PYTHON:-}" || { echo 'PDF migration blocked: set RAGY_PDF_PYTHON; see docs/pdf-go-feasibility.md' >&2; exit 1; }
	@command -v "$${RAGY_PDF_PYTHON}" >/dev/null || { echo 'Retained PDF interpreter is unavailable' >&2; exit 1; }

examples-project:
	@for module in $(filter examples/%,$(MODULES)); do \
		printf '\n[build] %s\n' "$$module"; \
		(cd "$$module" && $(GO) build ./...); \
	done
	@$(GO) build -o /dev/null ./examples/local-bm25

check-project:
	@printf '\n[integration] PostgreSQL\n'
	@./scripts/postgres-test.sh
	@printf '\n[integration] module artifacts and consumers\n'
	@cd tooling && $(GO) test -race -count=1 -timeout=30m -tags=integration ./...
	@$(MAKE) --no-print-directory test-pdf

test-pdf:
	@printf '\n[integration] PDF\n'
	@test -n "$${RAGY_PDF_PYTHON:-}" || { echo 'PDF migration blocked: set RAGY_PDF_PYTHON; see docs/pdf-go-feasibility.md' >&2; exit 1; }
	@cd adapters/pdf && $(GO) test -race -count=1 -run '^TestActualPDF' ./...

# Inputs are supplied by the release transaction, always in its isolated checkout.
release-prepare-project:
	@$(MAKE) --no-print-directory release-check-project
	@bash scripts/release-checksums.sh

release-check-project:
	@cd tooling && RAGY_CANDIDATE="$(RELEASE_CANDIDATE_DIR)" RAGY_RELEASE_VERSION="$(RELEASE_VERSION)" RAGY_ARTIFACT_PROXY="$(RELEASE_ARTIFACT_DIR)" \
		$(GO) test -count=1 -timeout=20m -tags=integration -run '^TestReleaseArtifacts$$' .

release-published-project:
	@cd tooling && RAGY_PUBLISHED_VERSION="$(RELEASE_VERSION)" \
		$(GO) test -count=1 -timeout=20m -tags=integration -run '^TestPublishedRelease$$' .
