GO := go
PYTHON := python3
VERIFY := GO="$(GO)" $(PYTHON) scripts/verify.py

.PHONY: lint fix test acceptance examples test-examples bench fuzz cover versions release-patch release-break

lint:
	@$(VERIFY) lint

# Cached developer checks; use acceptance for a fresh, recorded review gate.
test:
	@$(VERIFY) test

acceptance:
	@$(VERIFY) acceptance

versions:
	@$(VERIFY) versions

examples test-examples:
	@$(VERIFY) examples

fix:
	@$(VERIFY) fix

bench:
	@$(VERIFY) bench

fuzz:
	@$(VERIFY) fuzz --fuzz-seconds=$(or $(FUZZ_SECONDS),30)

cover:
	@$(VERIFY) cover

release-patch: acceptance
	@./scripts/release.sh patch "$(RELEASE_SOURCE)"

release-break: acceptance
	@./scripts/release.sh break "$(RELEASE_SOURCE)"
