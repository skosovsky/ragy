//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

type inventoryVerifier struct {
	calls    int
	mismatch bool
}

func (v *inventoryVerifier) VerifyInventory(
	_ context.Context,
	input lifecycle.Inventory,
) (lifecycle.InventoryConfirmation, error) {
	v.calls++
	fingerprint, err := input.Fingerprint()
	if err != nil {
		return lifecycle.InventoryConfirmation{}, err
	}
	if v.mismatch {
		fingerprint = "not-the-observed-inventory"
	}
	// The port can retain/mutate its owned request without touching the import envelope.
	if len(input.Unmanaged) > 0 {
		input.Unmanaged[0].Key = "mutated-port-input"
	}
	return lifecycle.InventoryConfirmation{
		Namespace:   input.Namespace,
		Watermark:   input.Watermark,
		Fingerprint: fingerprint,
		Coverage:    input.Coverage,
	}, nil
}
func inventoryFixture(kind lifecycle.InventoryKind, watermark string) lifecycle.Inventory {
	return lifecycle.Inventory{
		Namespace: "n", Kind: kind, Watermark: watermark, Coverage: lifecycle.FullInventory,
		Targets: []string{"dense", "tensor"}, Manifests: []lifecycle.Manifest{manifestFixture()},
		Unmanaged: []lifecycle.UnmanagedRecord{{Target: "dense", Key: "legacy-unknown-record"}},
	}
}

func TestBootstrapDeltaKeepsAbsentSourceAndUnmanagedRecords(t *testing.T) {
	// Arrange: faq is already managed; incoming delta only describes policy.
	_, store, _, _ := executorFixture(t)
	faq := manifestFixture()
	faq.ID, faq.Key, faq.Identity.Source = "faq-publication", "faq-key", "faq"
	for i := range faq.Targets {
		for j := range faq.Targets[i].Artifacts {
			faq.Targets[i].Artifacts[j].Reference.Source = "faq"
		}
	}
	snapshot := lifecycle.Snapshot{Schema: lifecycle.SchemaIdentity, Namespace: "n",
		Manifests: []lifecycle.Manifest{faq}, Publications: []lifecycle.Publication{{Source: "faq", Manifest: faq.ID}},
	}
	if _, err := store.CompareSwap(context.Background(), 0, snapshot); err != nil {
		t.Fatal(err)
	}
	verifier := &inventoryVerifier{}
	bootstrap, err := lifecycle.NewBootstrapper(store, verifier)
	if err != nil {
		t.Fatal(err)
	}
	input := inventoryFixture(lifecycle.DeltaInventory, "delta-1")
	// Act.
	receipt, err := bootstrap.Import(context.Background(), input)
	// Assert: no physical mutation port exists here and no missing-source deletion inferred.
	if err != nil || len(receipt.Missing) != 0 || receipt.Unmanaged[0].Key != "legacy-unknown-record" {
		t.Fatal("delta/ownership failed", err)
	}
	loaded, err := store.Load(context.Background(), "n")
	if err != nil || len(loaded.Publications) != 2 || len(loaded.Manifests) != 2 {
		t.Fatal("faq removed or unknown record adopted", err)
	}
	input.Unmanaged[0].Key = "mutated-caller"
	receipt.Unmanaged[0].Key = "mutated-output"
	loaded, err = store.Load(context.Background(), "n")
	if err != nil || loaded.Inventories[0].Unmanaged[0].Key != "legacy-unknown-record" {
		t.Fatal("receipt not owned", err)
	}
	if _, err = bootstrap.Import(
		context.Background(),
		inventoryFixture(lifecycle.DeltaInventory, "delta-1"),
	); err != nil ||
		verifier.calls != 1 {
		t.Fatal("idempotent import verified/wrote again", err)
	}
	changed := inventoryFixture(lifecycle.DeltaInventory, "delta-1")
	changed.Unmanaged[0].Key = "different"
	if _, err = bootstrap.Import(context.Background(), changed); !errors.Is(err, lifecycle.ErrIdempotencyConflict) {
		t.Fatal("watermark reused for different inventory")
	}
}

func TestBootstrapCompleteProposalsStayPinnedAcrossReplay(t *testing.T) {
	// Arrange: initialize policy and faq, then complete inventory omits faq.
	_, store, _, _ := executorFixture(t)
	verifier := &inventoryVerifier{}
	bootstrap, err := lifecycle.NewBootstrapper(store, verifier)
	if err != nil {
		t.Fatal(err)
	}
	initial := inventoryFixture(lifecycle.DeltaInventory, "initial")
	faq := manifestFixture()
	faq.ID, faq.Key, faq.Identity.Source = "faq1", "faq-key1", "faq"
	for i := range faq.Targets {
		for j := range faq.Targets[i].Artifacts {
			faq.Targets[i].Artifacts[j].Reference.Source = "faq"
		}
	}
	initial.Manifests = append(initial.Manifests, faq)
	if _, err = bootstrap.Import(context.Background(), initial); err != nil {
		t.Fatal(err)
	}
	complete := inventoryFixture(lifecycle.CompleteInventory, "complete-1")
	// Act.
	receipt, err := bootstrap.Import(context.Background(), complete)
	// Assert: missing source is a captured proposal, not a tombstone/physical deletion.
	if err != nil || len(receipt.Missing) != 1 || receipt.Missing[0].Manifest != "faq1" {
		t.Fatal("complete inventory proposal missing", err)
	}
	snapshot, err := store.Load(context.Background(), "n")
	if err != nil || len(snapshot.Publications) != 2 {
		t.Fatal("complete import removed source", err)
	}
	// Arrange: faq advances after the old watermark; replay must not target faq2.
	newer := faq
	newer.ID, newer.Key, newer.ExpectedPublication, newer.Identity.Revision = "faq2", "faq-key2", "faq1", "r2"
	for i := range newer.Targets {
		newer.Targets[i].Revision = "r2"
		for j := range newer.Targets[i].Artifacts {
			newer.Targets[i].Artifacts[j].Reference.Revision = "r2"
		}
	}
	snapshot.Manifests = append(snapshot.Manifests, newer)
	for i := range snapshot.Publications {
		if snapshot.Publications[i].Source == "faq" {
			snapshot.Publications[i].Manifest = "faq2"
		}
	}
	if _, err = store.CompareSwap(context.Background(), snapshot.Generation, snapshot); err != nil {
		t.Fatal(err)
	}
	replay, err := bootstrap.Import(context.Background(), complete)
	if err != nil || replay.Missing[0].Manifest != "faq1" {
		t.Fatal("old complete watermark targeted a newer publication", err)
	}
}

func TestBootstrapInvalidCoverageAndUnconfirmedEnvelopeFailBeforeCommit(t *testing.T) {
	// Arrange.
	_, store, _, _ := executorFixture(t)
	verifier := &inventoryVerifier{}
	bootstrap, err := lifecycle.NewBootstrapper(store, verifier)
	if err != nil {
		t.Fatal(err)
	}
	invalid := inventoryFixture(lifecycle.CompleteInventory, "full")
	invalid.Coverage = lifecycle.PartialInventory
	// Act/Assert: complete+partial fails before verifier; unconfirmed digest never commits.
	if _, err = bootstrap.Import(
		context.Background(),
		invalid,
	); !errors.Is(err, ragy.ErrInvalidArgument) ||
		verifier.calls != 0 {
		t.Fatal("incomplete complete inventory admitted")
	}
	verifier.mismatch = true
	if _, err = bootstrap.Import(
		context.Background(),
		inventoryFixture(lifecycle.DeltaInventory, "delta"),
	); !errors.Is(
		err,
		ragy.ErrProtocol,
	) {
		t.Fatal("unconfirmed envelope imported")
	}
	loaded, err := store.Load(context.Background(), "n")
	if err != nil || loaded.Generation != 0 || len(loaded.Manifests) != 0 {
		t.Fatal("failed verification changed durable state", err)
	}
}
