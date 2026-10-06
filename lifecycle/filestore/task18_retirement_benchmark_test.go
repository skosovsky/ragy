//go:build darwin || linux

package filestore_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

// BenchmarkTask18Retirement compares the same confirmed-cleaned pair fixture
// before/after explicit maintenance in the current implementation. It is separate
// from the original active-publication baseline; it does not fabricate an old API.
func BenchmarkTask18Retirement(b *testing.B) {
	for _, history := range []int{10, 100, 1000} {
		for _, compacted := range []bool{false, true} {
			phase := "CASBeforeCompaction"
			if compacted {
				phase = "CASAfterCompaction"
			}
			b.Run(fmt.Sprintf("%s/Artifacts32/History%d", phase, history), func(b *testing.B) {
				runRetirementCASBenchmark(b, history, compacted)
			})
		}
	}
}

func runRetirementCASBenchmark(b *testing.B, history int, compacted bool) {
	store, err := filestore.New(b.TempDir(), 64<<20)
	if err != nil {
		b.Fatal(err)
	}
	snapshot, ids := retirementBenchmarkFixture(history)
	snapshot, err = store.CompareSwap(context.Background(), 0, snapshot)
	if err != nil {
		b.Fatal(err)
	}
	if compacted {
		snapshot, err = store.Maintain(
			context.Background(),
			snapshot.Generation,
			lifecycle.RetirementRequest{Namespace: "n", Manifests: ids},
		)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		snapshot, err = store.CompareSwap(context.Background(), snapshot.Generation, snapshot)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	data, err := json.Marshal(snapshot)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportMetric(float64(len(data)), "snapshot-bytes")
}

func retirementBenchmarkFixture(n int) (lifecycle.Snapshot, []string) {
	snapshot := lifecycle.Snapshot{Schema: lifecycle.SchemaIdentity, Namespace: "n"}
	ids := make([]string, 0, n)
	for i := range n {
		old := fixture().Manifests[0]
		old.ID, old.Key, old.Identity.Source = fmt.Sprintf(
			"old-%d",
			i,
		), fmt.Sprintf(
			"old-key-%d",
			i,
		), fmt.Sprintf(
			"source-%d",
			i,
		)
		first := old.Targets[0].Artifacts[0]
		first.Reference.Source = old.Identity.Source
		old.Targets[0].Artifacts = nil
		for j := range 32 {
			artifact := first
			artifact.Reference.Artifact = fmt.Sprintf("artifact-%d", j)
			artifact.Supports = nil
			artifact.Supports = append(artifact.Supports, artifact.Reference)
			old.Targets[0].Artifacts = append(old.Targets[0].Artifacts, artifact)
		}
		owner := old
		owner.ID, owner.Key, owner.ExpectedPublication = fmt.Sprintf(
			"owner-%d",
			i,
		), fmt.Sprintf(
			"owner-key-%d",
			i,
		), old.ID
		owner.Targets = nil
		owner.Tombstone = true
		owner.State = lifecycle.Complete
		snapshot.Manifests = append(snapshot.Manifests, old, owner)
		snapshot.Publications = append(
			snapshot.Publications,
			lifecycle.Publication{Source: owner.Identity.Source, Manifest: owner.ID},
		)
		snapshot.Cleanups = append(
			snapshot.Cleanups,
			lifecycle.CleanupJob{
				Owner:     owner.ID,
				StartedAt: owner.PublishedAt,
				Deadline:  owner.PublishedAt.Add(time.Hour),
				Complete:  true,
				Items: []lifecycle.RetiredTarget{
					{
						Manifest: old.ID,
						Target:   "lexical",
						State:    lifecycle.CleanupDone,
						Attempts: 1,
						NextAt:   owner.PublishedAt,
					},
				},
			},
		)
		ids = append(ids, old.ID)
	}
	return snapshot, ids
}
