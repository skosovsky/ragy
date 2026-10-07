//go:build integration_pg

package pgvector

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func TestRealPostgresScopedTenantPairsAndPinnedAdmission(t *testing.T) {
	// Arrange: actual records for adjacent int64 tenant identities and an omitted one.
	container := os.Getenv("RAGY_PG_TEST_CONTAINER")
	if container == "" {
		t.Fatal("integration_pg requires RAGY_PG_TEST_CONTAINER; no SKIP")
	}
	label, err := exec.CommandContext(t.Context(), "docker", "inspect", "--format", `{{index .Config.Labels "ragy.task20"}}`, container).CombinedOutput()
	if err != nil || strings.TrimSpace(string(label)) != "T09" {
		t.Fatal("isolated profile label required", err)
	}
	db := &psqlDB{container: container}
	table := fmt.Sprintf("ragy_t17_scope_%d", os.Getpid())
	if _, err = db.command(t.Context(), "CREATE TABLE "+table+" (id text PRIMARY KEY, content text NOT NULL, attributes jsonb NOT NULL, vector vector(1) NOT NULL);"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := db.command(ctx, "DROP TABLE "+table+";"); err != nil {
			t.Error(err)
		}
	})
	fields := filter.NewSchema()
	tenant, err := fields.Int("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	codec := &canonicalCodec{}
	store, err := New(db, Config[integerWireMeta]{Table: table, Schema: schema, Space: fixtureSpace()}, codec)
	if err != nil {
		t.Fatal(err)
	}
	records := []dense.Record[integerWireMeta]{
		{ID: "a", Content: "tenant a", Vector: []float32{1}, Space: fixtureSpace(), Meta: integerWireMeta{Tenant: 9007199254740992}},
		{ID: "b", Content: "tenant b", Vector: []float32{1}, Space: fixtureSpace(), Meta: integerWireMeta{Tenant: 9007199254740993}},
	}
	if err = store.Upsert(t.Context(), records); err != nil {
		t.Fatal(err)
	}
	if _, err = db.command(t.Context(), "INSERT INTO "+table+" VALUES ('missing','text','{}','[1]');"); err != nil {
		t.Fatal(err)
	}
	conditions := []filter.Condition{nativeTenantCondition(t, schema, tenant, 9007199254740992), nativeTenantCondition(t, schema, tenant, 9007199254740993)}
	for i, condition := range conditions {
		read := nativeTenantRead(t, schema, condition)
		// Act.
		result, queryErr := store.Retrieve(t.Context(), retrieval.Query[struct{}]{Read: read, Options: retrieval.RetrieveOptions{Space: fixtureSpace(), Vector: []float32{1}, TopK: 3}})
		// Assert: mandatory binding reaches SQL; neither neighbor nor omitted tenant leaks.
		if queryErr != nil || result.Len() != 1 || result.Documents()[0].ID != records[i].ID {
			t.Fatal("native tenant pair", i, result, queryErr)
		}
	}
	read := nativeTenantRead(t, schema, conditions[0])
	// Act: optional metadata cannot replace the mandatory tenant.
	conflict, conflictErr := store.Retrieve(t.Context(), retrieval.Query[struct{}]{Read: read, Options: retrieval.RetrieveOptions{Space: fixtureSpace(), Vector: []float32{1}, TopK: 3, Filters: conditions[1]}})
	// Assert.
	if conflictErr != nil || conflict.Len() != 0 {
		t.Fatal("mandatory/optional intersection", conflict, conflictErr)
	}
	checkNativePinnedAdmission(t, db, store)
	t.Log("native Binding tenant pairs >2^53, omission, conflicting optional predicate and pinned-before-I/O verified")
}

func nativeTenantCondition(t *testing.T, schema filter.Schema, field filter.Field[int64], value int64) filter.Condition {
	t.Helper()
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := filter.Eq(builder, field, value).Build()
	if err != nil {
		t.Fatal(err)
	}
	return condition
}

func nativeTenantRead(t *testing.T, schema filter.Schema, mandatory filter.Condition) access.Binding {
	t.Helper()
	now := time.Unix(1000, 0)
	read, err := access.Scoped(access.ScopedConfig{Schema: schema, Mandatory: mandatory, Publication: access.CurrentPublication(), Snapshot: access.Snapshot{Identity: "test-host-decision", PolicyEpoch: 1, IssuedAt: now, ExpiresAt: now.Add(time.Hour)}, Now: func() time.Time { return now }, Authority: access.AuthorityFunc(func(ctx context.Context, _ access.Snapshot) error { return ctx.Err() })})
	if err != nil {
		t.Fatal(err)
	}
	return read
}

func checkNativePinnedAdmission(t *testing.T, db *psqlDB, store *Store[integerWireMeta]) {
	t.Helper()
	publication, err := access.PinPublication("pinned", nil)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	before := db.calls
	result, err := store.Retrieve(t.Context(), retrieval.Query[struct{}]{Read: read, Options: retrieval.RetrieveOptions{Space: fixtureSpace(), Vector: []float32{1}, TopK: 3}})
	if !errors.Is(err, ragy.ErrUnsupported) || !access.IsProtectionFailure(err) || result.Len() != 0 || db.calls != before {
		t.Fatal("unsupported pin reached native DB", result, err, db.calls, before)
	}
}
