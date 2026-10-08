package bridge_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"testing"

	"github.com/skosovsky/memy"

	bridge "github.com/skosovsky/ragy/examples/context-bridge"

	"github.com/skosovsky/ragy/retrieval"
)

func TestAdversarialEvidenceCorruption(t *testing.T) {
	for _, fault := range []string{"remove-snippets", "remove-lineage", "wrong-contributor", "rewrite-provenance"} {
		t.Run(fault, func(t *testing.T) {
			// Arrange: valid canonical snapshot.
			f := newFixture(t)
			out, err := f.b.Run(t.Context())
			check(t, err)
			var wire map[string]json.RawMessage
			check(t, json.Unmarshal(out.Durable, &wire))
			var exts []map[string]json.RawMessage
			check(t, json.Unmarshal(wire["extensions"], &exts))
			var side bridge.Sidecar[uncertainty]
			check(t, json.Unmarshal(exts[0]["payload"], &side))
			switch fault {
			case "remove-snippets":
				side.Snippets = nil
			case "remove-lineage":
				side.Snippets = nil
				side.Inputs = nil
				side.References = nil
			case "wrong-contributor":
				side.Snippets[0].Contributors[0].InputIndex = 1
			case "rewrite-provenance":
				side.Snippets[0].DocumentID = "record-1"
			}
			exts[0]["payload"], err = json.Marshal(side)
			check(t, err)
			wire["extensions"], err = json.Marshal(exts)
			check(t, err)
			raw, err := json.Marshal(wire)
			check(t, err)
			// Act: decode a deliberately corrupted snapshot.
			decoded, err := bridge.Decode[uncertainty](
				t.Context(),
				raw,
				bridge.Registry[uncertainty]("host.uncertainty/1"),
			)
			if err == nil && fault == "remove-lineage" {
				sink, e := bridge.NewSnapshotSink[uncertainty]("test", "host.uncertainty/1")
				check(t, e)
				check(t, sink.Publish(t.Context(), decoded))
				_, e = sink.Purge(
					t.Context(),
					memy.PurgeBatch{Scope: f.b.Scope, Records: []string{"record-0", "record-1"}},
				)
				check(t, e)
				if len(sink.Snapshots()) != 0 {
					t.Fatalf(
						"accepted stripped canonical lineage: Purge acknowledged but %d snapshots retained",
						len(sink.Snapshots()),
					)
				}
			}
			// Assert: the mandatory evidence must remain associated.
			if err == nil {
				t.Fatalf(
					"accepted corruption %s with %d refs / %d snippets and nonempty text=%q",
					fault,
					len(decoded.Evidence.References),
					len(decoded.Evidence.Snippets),
					decoded.Evidence.Text,
				)
			}
		})
	}
}
func TestAdversarialLostPolicyCause(t *testing.T) {
	// Arrange: valid canonical snapshot.
	f := newFixture(t)
	f.b.Retrieve = func(context.Context) (bridge.Batch[metadata], error) {
		return bridge.Batch[metadata]{}, memy.ErrPolicyDenied
	}
	// Act.
	_, err := f.b.Run(t.Context())
	// Assert.
	if !errors.Is(err, memy.ErrPolicyDenied) {
		t.Fatalf("lost safe stable cause: %v", err)
	}
}

func TestRelationalValidationWithRecomputedChecksum(t *testing.T) {
	for _, fault := range []string{"contributor", "canonical-id", "missing-snippets", "native-score", "duplicate-snippet", "false-completeness", "contributor-completeness", "truncated-precision"} {
		t.Run(fault, func(t *testing.T) {
			// Arrange: recomputation removes checksum as an explanation for rejection.
			f := newFixture(t)
			out, err := f.b.Run(t.Context())
			check(t, err)
			var wire map[string]json.RawMessage
			check(t, json.Unmarshal(out.Durable, &wire))
			var extensions []map[string]json.RawMessage
			check(t, json.Unmarshal(wire["extensions"], &extensions))
			var side bridge.Sidecar[uncertainty]
			check(t, json.Unmarshal(extensions[0]["payload"], &side))
			switch fault {
			case "contributor":
				side.Snippets[0].Contributors[0].InputIndex = 1
			case "canonical-id":
				side.Snippets[0].DocumentID = "record-1"
			case "missing-snippets":
				side.Snippets = nil
			case "native-score":
				side.Snippets[0].Rank++
			case "duplicate-snippet":
				side.Snippets[1] = side.Snippets[0]
			case "false-completeness", "contributor-completeness":
				side.Precision = "unavailable"
				side.Truncated = true
				for i := range side.Snippets {
					side.Snippets[i].Span = nil
					side.Snippets[i].FullDocument = false
					side.Snippets[i].DeliveryUncertain = true
					side.Snippets[i].Contributors[0].FullDocument = false
					side.Snippets[i].Contributors[0].DeliveryUncertain = true
				}
				if fault == "false-completeness" {
					side.Snippets[0].FullDocument = true
					side.Snippets[0].DeliveryUncertain = false
				} else {
					side.Snippets[0].Contributors[0].FullDocument = true
					side.Snippets[0].Contributors[0].DeliveryUncertain = false
				}
			case "truncated-precision":
				side.Truncated = true
			}
			side.Digest = ""
			raw, err := json.Marshal(side)
			check(t, err)
			digest := sha256.Sum256(raw)
			side.Digest = hex.EncodeToString(digest[:])
			extensions[0]["payload"], err = json.Marshal(side)
			check(t, err)
			wire["extensions"], err = json.Marshal(extensions)
			check(t, err)
			raw, err = json.Marshal(wire)
			check(t, err)
			// Act.
			_, err = bridge.Decode[uncertainty](t.Context(), raw, bridge.Registry[uncertainty]("host.uncertainty/1"))
			// Assert: actual associations and inventory, not only checksums, reject corruption.
			if err == nil {
				t.Fatal("relational corruption accepted")
			}
		})
	}
}

func TestAdversarialUnrankedDocuments(t *testing.T) {
	// Arrange: native zero is valid, and must stay distinct from the renderer's fallback rank.
	f := newFixture(t)
	for i := range f.batch.Documents {
		f.batch.Documents[i].Rank = 0
	}
	// Act: exercise canonical recall, render, encode and decode.
	out, err := f.b.Run(t.Context())
	check(t, err)
	decoded, err := bridge.Decode[uncertainty](
		t.Context(), out.Durable, bridge.Registry[uncertainty]("host.uncertainty/1"),
	)
	check(t, err)
	// Assert: native evidence is retained while the rendered order is explicit.
	for i, input := range decoded.Evidence.Inputs {
		if input.Rank != 0 || decoded.Evidence.Snippets[i].Rank != i+1 {
			t.Fatal("native rank replaced or renderer fallback rejected")
		}
	}
}

func TestCanonicalUncertaintyWithoutHostExtension(t *testing.T) {
	// Arrange: typed host uncertainty is optional; known canonical evidence is not.
	f := newFixture(t)
	f.b.Uncertainty = nil
	// Act: Run includes the real durable codec roundtrip.
	out, err := f.b.Run(t.Context())
	check(t, err)
	// Assert: canonical loss/uncertainty survives independently of the optional host type.
	for _, input := range out.Evidence.Inputs {
		if input.Uncertainty != nil || input.Extractor != "host/1" ||
			len(input.Losses) != 1 || input.Losses[0] != "canonical-loss" ||
			len(input.Uncertainties) != 1 || input.Uncertainties[0] != "canonical-uncertainty" {
			t.Fatal("canonical extractor evidence lost without optional host extension")
		}
	}
}

func TestAdversarialRendererBudgetCause(t *testing.T) {
	// Arrange: the renderer must reject a final envelope larger than its hard byte budget.
	f := newFixture(t)
	f.b.Options.Render.Resource.MaxOutputBytes = 1
	// Act.
	_, err := f.b.Run(t.Context())
	// Assert: preserve the stable cause without publishing partial output.
	if !errors.Is(err, retrieval.ErrArtifactLimit) || len(f.sink.data) != 0 {
		t.Fatalf("renderer budget failure lost classification or published: %v", err)
	}
}

func TestManagedSinkRejectsLatePublishAndPreservesNewEpoch(t *testing.T) {
	// Arrange: real canonical output and a separate managed sink.
	f := newFixture(t)
	old, err := f.b.Run(t.Context())
	check(t, err)
	sink, err := bridge.NewSnapshotSink[uncertainty]("epoch-context", "host.uncertainty/1")
	check(t, err)
	check(t, sink.Publish(t.Context(), old))
	batch := memy.PurgeBatch{
		Scope:       f.b.Scope,
		OperationID: "old-purge",
		Epoch:       1,
		Records:     []string{"record-0", "record-1"},
	}
	// Act: revoke sink lineage, then try to publish the queued old serialized output.
	_, err = sink.Purge(t.Context(), batch)
	check(t, err)
	lateErr := sink.Publish(t.Context(), old)
	// Assert: the sink's own fence prevents late replay.
	if !errors.Is(lateErr, memy.ErrRevoked) || len(sink.Snapshots()) != 0 {
		t.Fatal("late sink publication bypassed epoch")
	}
	// Arrange: advance the actual canonical scope epoch while retaining record-1.
	_, err = f.forget(t.Context())
	check(t, err)
	f.batch.Documents = f.batch.Documents[1:]
	f.b.Publish = sink.Publish
	// Act: publish eligible canonical data under the new epoch, then replay the older sink batch.
	current, err := f.b.Run(t.Context())
	check(t, err)
	_, err = sink.Purge(t.Context(), batch)
	check(t, err)
	// Assert: an older/equal-epoch purge cannot delete newly admitted data.
	if current.Evidence.Epoch != batch.Epoch || len(sink.Snapshots()) != 1 {
		t.Fatal("old cleanup deleted current epoch output")
	}
}
