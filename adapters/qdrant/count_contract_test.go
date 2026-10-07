package qdrant

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/filter"
)

type countClient struct {
	*fakeClient

	count int
	calls int
}

func (c *countClient) DeleteByIDs(context.Context, string, []string) (int, error) {
	c.calls++
	return c.count, nil
}
func (c *countClient) DeleteByFilter(context.Context, string, Condition) (int, error) {
	c.calls++
	return c.count, nil
}

func TestDeletionRequiresExactNonnegativeCounts(t *testing.T) {
	// Arrange: a bridge's -1 unknown sentinel must not become success.
	fields := filter.NewSchema()
	age, err := fields.Int("age")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := filter.Eq(builder, age, int64(7)).Build()
	if err != nil {
		t.Fatal(err)
	}
	for _, count := range []int{-1, 0, 5} {
		client := &countClient{fakeClient: &fakeClient{}, count: count}
		store, err := New(
			client,
			Config[contracttest.StructMeta]{Collection: "docs", Space: fixtureSpace(), Schema: schema},
			contracttest.JSONCodec[contracttest.StructMeta](t, schema),
		)
		if err != nil {
			t.Fatal(err)
		}
		// Act.
		byIDs, idErr := store.DeleteByIDs(t.Context(), []string{"one"})
		byFilter, filterErr := store.DeleteByFilter(t.Context(), condition)
		// Assert: counts come from the bridge, not from submitted ID length.
		if client.calls != 2 {
			t.Fatal("unexpected callback count", client.calls)
		}
		if count < 0 {
			if !errors.Is(idErr, ragy.ErrProtocol) || !errors.Is(filterErr, ragy.ErrProtocol) || byIDs.Deleted != 0 ||
				byFilter.Deleted != 0 {
				t.Fatal("unknown count accepted", byIDs, idErr, byFilter, filterErr)
			}
		} else if idErr != nil || filterErr != nil || byIDs.Deleted != count || byFilter.Deleted != count {
			t.Fatal("exact count changed", count, byIDs, idErr, byFilter, filterErr)
		}
	}
}
