//go:build darwin || linux

package filestore_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

// Each retained manifest has one source publication and one supported artifact.
// CAS mutates no history count; its cost is measured at a fixed retained size.
//
//nolint:gocognit // Retained inventory fixture and timed durable CAS remain explicit.
func BenchmarkTask18Filestore(b *testing.B) {
	for _, artifacts := range []int{1, 32} {
		for _, n := range []int{10, 100, 1000} {
			b.Run(fmt.Sprintf("CAS/Artifacts%d/History%d", artifacts, n), func(b *testing.B) {
				//nolint:usetesting // Fixed local filesystem root is part of the benchmark profile.
				root, err := os.MkdirTemp("/tmp", "ragy-task18-")
				if err != nil {
					b.Fatal(err)
				}
				b.Cleanup(func() { _ = os.RemoveAll(root) })
				store, err := filestore.New(root, 64<<20)
				if err != nil {
					b.Fatal(err)
				}
				snapshot := lifecycle.Snapshot{Schema: lifecycle.SchemaIdentity, Namespace: "n"}
				for i := range n {
					row := fixture()
					m := row.Manifests[0]
					m.ID = fmt.Sprintf("publication-%d", i)
					m.Key = fmt.Sprintf("request-%d", i)
					m.Identity.Source = fmt.Sprintf("source-%d", i)
					m.Targets[0].Artifacts[0].Reference.Source = m.Identity.Source
					m.Targets[0].Artifacts[0].Supports[0].Source = m.Identity.Source
					first := m.Targets[0].Artifacts[0]
					for j := range artifacts - 1 {
						a := first
						a.Reference.Artifact = fmt.Sprintf("artifact-%d", j+1)
						a.Supports = nil
						a.Supports = append(a.Supports, a.Reference)
						m.Targets[0].Artifacts = append(m.Targets[0].Artifacts, a)
					}
					snapshot.Manifests = append(snapshot.Manifests, m)
					snapshot.Publications = append(
						snapshot.Publications,
						lifecycle.Publication{Source: m.Identity.Source, Manifest: m.ID},
					)
				}
				snapshot, err = store.CompareSwap(context.Background(), 0, snapshot)
				if err != nil {
					b.Fatal(err)
				}
				encoded, err := json.Marshal(snapshot)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(float64(len(encoded)), "snapshot-bytes")
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					snapshot, err = store.CompareSwap(context.Background(), snapshot.Generation, snapshot)
					if err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				encoded, err = json.Marshal(snapshot)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(float64(len(encoded)), "snapshot-bytes")
			})
		}
	}
}
