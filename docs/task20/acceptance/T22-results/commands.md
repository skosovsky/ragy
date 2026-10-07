# Actual final commands / exits

All command processes observed completed with exit0 except explicitly retained failed attempts.

- From root, PATH=/opt/homebrew/Cellar/go/1.26.5/bin:/opt/homebrew/bin:/usr/bin:/bin GOTOOLCHAIN=local GOWORK=off GOCACHE=/tmp/ragy-t20-go-cache GOMODCACHE=/tmp/ragy-t21-modcache GOPATH=/tmp/ragy-t21-gopath GOLANGCI_LINT_CACHE=/tmp/ragy-t20-lint-cache PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify.py acceptance. Recorded commands/exits in all-module log.
- Same selected Go/Python environment: python3 scripts/check_release_consumer.py fc2040c45a1ae50d147f801ff4b8f42c50044db6. Fresh task-owned temporary consumer/cache/proxy/localbare created by helper, caller unchanged.
- docker run --rm -d --name ragy-task20-t22-pg --label ragy.task20=T09 --label ragy.acceptance=T22 -e POSTGRES_PASSWORD=ragy-task20-fixture pgvector/pgvector:pg17 (image151aa1c2e849). After explicit CREATE DATABASE ragy and CREATE EXTENSION vector in ragy, from adapters/pgvector with /usr/local/bin additionally in PATH and RAGY_PG_TEST_CONTAINER=ragy-task20-t22-pg: go test -count=1 -race -tags=integration_pg -run '^TestRealPostgres' -v ./.... No host ports/mounts.
- From adapters/pdf, same standalone Go/caches, RAGY_PDF_PYTHON=/Users/skosovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3: go test -count=1 -race -v ./.... Same interpreter executes testdata/verify_fixture.py and testdata/verify_engine_errors.py.
- git diff --check and git show --format= --check HEAD exit0.

Default runner paid/actual opt-in tests may skip internally; overall unit PASS does not certify those profiles. Required actual PG/PDF separate commands have zero SKIPs. Actual runtime version/fixtures are in their logs. macOS local toolchain; no remote GitHub run/public release.
