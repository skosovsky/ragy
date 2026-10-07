GO := go
PYTHON := python3
VERIFY := GO="$(GO)" $(PYTHON) scripts/verify.py

.PHONY: check check-linux check-plan test-fast lint fix test acceptance examples test-examples bench fuzz cover versions release-patch release-break

lint:
	@$(VERIFY) lint

# All required lanes, including integration and consumers.
test:
	@$(VERIFY) test

check acceptance:
	@$(VERIFY) check

check-plan:
	@$(VERIFY) plan

check-linux:
	@$(PYTHON) scripts/check_linux.py check

test-fast:
	@$(VERIFY) test-fast

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

release-patch:
	@./scripts/release.sh patch "$(RELEASE_SOURCE)"

release-break:
	@./scripts/release.sh break "$(RELEASE_SOURCE)"
