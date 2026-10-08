//go:build integration

package pgvector

import (
	"context"
	"crypto/rand"
	"os/exec"
	"strings"
	"testing"
	"time"
)

const postgresTestImage = "pgvector/pgvector@sha256:ac08538c6f8b9904c33c8224c5e5706dbe760aca29db1d096972b4052c22a75d"

func postgresCommand(ctx context.Context, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "docker", args...)
	cmd.WaitDelay = 5 * time.Second
	return cmd.CombinedOutput()
}

func newPostgresDB(t *testing.T) *psqlDB {
	t.Helper()
	if out, err := postgresCommand(t.Context(), "info"); err != nil {
		t.Fatalf("integration_pg requires a running Docker daemon: %v\n%s", err, out)
	}
	name := "ragy-pg-" + strings.ToLower(rand.Text())
	started := false
	// Cleanup has its own context: t.Context is cancelled before t.Cleanup runs.
	// Register before run so a lost start response still cleans up our unique name.
	t.Cleanup(func() {
		out, err := postgresCommand(context.Background(), "rm", "-fv", name)
		if err != nil {
			if started {
				t.Errorf("PostgreSQL cleanup: %v\n%s", err, out)
			} else {
				t.Logf("PostgreSQL start cleanup: %v\n%s", err, out)
			}
		}
	})
	out, err := postgresCommand(t.Context(), "run", "-d", "--name", name, "--label", "ragy.test=postgres",
		"-e", "POSTGRES_PASSWORD=ragy-test", "-e", "POSTGRES_DB=ragy", postgresTestImage)
	if err != nil {
		t.Fatalf("start PostgreSQL: %v\n%s", err, out)
	}
	started = true
	ready, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		// The temporary initdb server exposes only a Unix socket. Wait for final TCP readiness.
		if _, err := postgresCommand(
			ready,
			"exec",
			name,
			"pg_isready",
			"-h",
			"127.0.0.1",
			"-U",
			"postgres",
			"-d",
			"ragy",
		); err == nil {
			break
		}
		state, err := postgresCommand(ready, "inspect", "--format", "{{.State.Running}}", name)
		if err != nil || strings.TrimSpace(string(state)) != "true" {
			logs, _ := postgresCommand(t.Context(), "logs", name)
			t.Fatalf("PostgreSQL readiness failed: %v\n%s\n%s", err, state, logs)
		}
		select {
		case <-ready.Done():
			logs, _ := postgresCommand(t.Context(), "logs", name)
			t.Fatalf("PostgreSQL readiness deadline: %v\n%s", ready.Err(), logs)
		case <-ticker.C:
		}
	}
	db := &psqlDB{container: name}
	if _, err := db.command(t.Context(), "CREATE EXTENSION IF NOT EXISTS vector;"); err != nil {
		t.Fatal(err)
	}
	return db
}
