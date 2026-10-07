//go:build integration_pg

package pgvector

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/retrieval"
)

func TestRealPostgresCanonicalCodecAndQuotedTable(t *testing.T) {
	// Arrange: actual mixed-case table and a codec that rejects raw json.Number.
	container := os.Getenv("RAGY_PG_TEST_CONTAINER")
	if container == "" {
		t.Fatal("integration_pg requires RAGY_PG_TEST_CONTAINER; no SKIP")
	}
	label, err := exec.CommandContext(t.Context(), "docker", "inspect", "--format", `{{index .Config.Labels "ragy.task20"}}`, container).
		CombinedOutput()
	if err != nil || strings.TrimSpace(string(label)) != "T09" {
		t.Fatal("isolated profile label required", err)
	}
	db := &psqlDB{container: container}
	table := fmt.Sprintf("RagyT17_%d", os.Getpid())
	quoted := `"` + table + `"`
	if _, err = db.command(
		t.Context(),
		"CREATE TABLE "+quoted+" (id text PRIMARY KEY, content text NOT NULL, attributes jsonb NOT NULL, vector vector(1) NOT NULL);",
	); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := db.command(ctx, "DROP TABLE "+quoted+";"); err != nil {
			t.Error(err)
		}
	})
	codec := &canonicalCodec{}
	store, err := New(
		db,
		Config[integerWireMeta]{Table: table, Schema: integerWireSchema(t), Space: fixtureSpace()},
		codec,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err = store.Upsert(
		t.Context(),
		[]dense.Record[integerWireMeta]{
			{
				ID:      "precise",
				Content: "text",
				Vector:  []float32{1},
				Space:   fixtureSpace(),
				Meta:    integerWireMeta{Tenant: 9007199254740993},
			},
		},
	); err != nil {
		t.Fatal(err)
	}
	if _, err = db.command(t.Context(), "INSERT INTO "+quoted+" VALUES ('empty','text','{}','[1]');"); err != nil {
		t.Fatal(err)
	}
	// Act.
	result, queryErr := retrieveStore(
		t.Context(),
		store,
		"",
		retrieval.RetrieveOptions{TopK: 2, Space: fixtureSpace(), Vector: []float32{1}},
	)
	found, findErr := store.FindByIDs(t.Context(), []string{"precise", "empty"})
	deleted, deleteErr := store.DeleteByIDs(t.Context(), []string{"precise"})
	// Assert: query and raw administration use the same canonical codec boundary.
	if queryErr != nil || findErr != nil || deleteErr != nil || result.Len() != 2 || len(found) != 2 ||
		codec.calls != 4 ||
		deleted.Deleted != 1 {
		t.Fatal("native bridge", result, queryErr, found, findErr, deleted, deleteErr, codec.calls)
	}
	for _, doc := range result.Documents() {
		want := int64(0)
		if doc.ID == "precise" {
			want = 9007199254740993
		}
		if doc.Meta.Tenant != want {
			t.Fatal("native integer/empty identity", doc)
		}
	}
	t.Logf(
		"native quoted table=%s, canonical custom codec calls=%d, exact deleted=%d",
		table,
		codec.calls,
		deleted.Deleted,
	)
}
