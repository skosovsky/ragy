//go:build darwin || linux

package managed

import (
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

func TestManagedOwnsBM25ParametersAcrossSnapshotBuilds(t *testing.T) {
	// Arrange: constructor must detach configuration retained for future builds.
	store, err := filestore.New(t.TempDir(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	parameters := lexical.BM25Parameters{K1: 0, B: 0}
	adapter, err := New(
		Config[struct{}]{
			Namespace:          "n",
			Target:             "lexical",
			Store:              store,
			Schema:             filter.EmptySchema(),
			BM25:               lexical.Config[struct{}]{SearchFields: []string{"content"}, Parameters: &parameters},
			CloneMeta:          func(m struct{}) (struct{}, error) { return m, nil },
			MaxCachedSnapshots: 1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act: build using retained configuration after caller mutation.
	parameters.K1, parameters.B = 99, 99
	_, err = lexical.NewBM25Index(adapter.config.Schema, adapter.config.BM25, nil, nil)
	// Assert: future snapshot build preserves explicit zeros and stays valid.
	if err != nil || adapter.config.BM25.Parameters == &parameters ||
		*adapter.config.BM25.Parameters != (lexical.BM25Parameters{}) {
		t.Fatal("borrowed managed parameters", adapter.config.BM25.Parameters, err)
	}
}
