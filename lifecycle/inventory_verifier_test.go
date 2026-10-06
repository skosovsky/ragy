package lifecycle_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

type observedTarget struct {
	name     string
	active   *[]string
	calls    int
	behavior string
}

func (o *observedTarget) InventoryTarget() string { return o.name }
func (o *observedTarget) ObserveInventory(_ context.Context, input lifecycle.Inventory, next func() error) error {
	o.calls++
	input.Unmanaged[0].Key = "mutated-observer-input"
	*o.active = append(*o.active, o.name)
	defer func() { *o.active = (*o.active)[:len(*o.active)-1] }()
	switch o.behavior {
	case "missing":
		return nil
	case "error":
		return ragy.ErrUnavailable
	case "twice":
		_ = next()
		_ = next()
		return nil
	case "swallow":
		_ = next()
		return nil
	default:
		if o.name == "tensor" && !slices.Equal(*o.active, []string{"dense", "tensor"}) {
			return ragy.ErrProtocol
		}
		return next()
	}
}

func TestFencedVerifierOwnsConfigurationAndObservesTogether(t *testing.T) {
	// Arrange: reverse input order and mutable host observer registration map.
	active := []string{}
	dense := &observedTarget{name: "dense", active: &active}
	tensor := &observedTarget{name: "tensor", active: &active}
	ports := map[string]lifecycle.InventoryObserver{"tensor": tensor, "dense": dense}
	verifier, err := lifecycle.NewFencedInventoryVerifier(ports)
	if err != nil {
		t.Fatal(err)
	}
	delete(ports, "dense")
	input := inventoryFixture(lifecycle.CompleteInventory, "fenced")
	input.Targets = []string{"tensor", "dense"}
	fingerprint, err := input.Fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	// Act: input/config mutation by one observer cannot alter the captured envelope.
	confirmed, err := verifier.VerifyInventory(t.Context(), input)
	// Assert: exact original digest and simultaneous deterministic fences.
	if err != nil || confirmed.Fingerprint != fingerprint || confirmed.Watermark != "fenced" ||
		confirmed.Coverage != lifecycle.FullInventory ||
		dense.calls != 1 ||
		tensor.calls != 1 ||
		len(active) != 0 {
		t.Fatal("unfenced confirmation", confirmed, err)
	}
	input.Targets = []string{"dense"}
	input.Manifests = nil
	if _, err = verifier.VerifyInventory(t.Context(), input); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal("incomplete target profile", err)
	}
	if dense.calls != 1 || tensor.calls != 1 {
		t.Fatal("unsupported observer executed")
	}
}

func TestFencedVerifierRejectsMissingDuplicateAndSwallowedFailure(t *testing.T) {
	for _, scenario := range []struct {
		name, dense, tensor string
		want                error
	}{
		{name: "missing", dense: "missing", want: ragy.ErrProtocol},
		{name: "duplicate", dense: "twice", want: ragy.ErrProtocol},
		{name: "swallowed", dense: "swallow", tensor: "error", want: ragy.ErrUnavailable},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			// Arrange.
			active := []string{}
			verifier, err := lifecycle.NewFencedInventoryVerifier(map[string]lifecycle.InventoryObserver{
				"dense":  &observedTarget{name: "dense", active: &active, behavior: scenario.dense},
				"tensor": &observedTarget{name: "tensor", active: &active, behavior: scenario.tensor},
			})
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			confirmed, err := verifier.VerifyInventory(
				t.Context(),
				inventoryFixture(lifecycle.CompleteInventory, "bad"),
			)
			// Assert: no confirmation escapes a missing fence or a failed nested target.
			if !errors.Is(err, scenario.want) || confirmed.Fingerprint != "" {
				t.Fatal("invalid confirmation", confirmed, err)
			}
		})
	}
}
