package lifecycle

import (
	"context"
	"slices"

	ragy "github.com/skosovsky/ragy"
)

// InventoryObserver verifies its target's exact retained data and holds a mutation
// fence while next executes synchronously once. It must not adopt or delete data.
// Complete coverage includes all opaque unmanaged keys; delta may omit other data.
type InventoryObserver interface {
	InventoryTarget() string
	ObserveInventory(context.Context, Inventory, func() error) error
}

// FencedInventoryVerifier observes all configured targets under simultaneous fences.
// The confirmation describes an observation point, not a lease or distributed commit.
type FencedInventoryVerifier struct {
	targets   []string
	observers map[string]InventoryObserver
}

func NewFencedInventoryVerifier(observers map[string]InventoryObserver) (*FencedInventoryVerifier, error) {
	if len(observers) == 0 {
		return nil, ragy.ErrInvalidArgument
	}
	owned := make(map[string]InventoryObserver, len(observers))
	names := make([]string, 0, len(observers))
	for name, observer := range observers {
		if !identities(name) || nilPort(observer) || observer.InventoryTarget() != name {
			return nil, ragy.ErrInvalidArgument
		}
		names = append(names, name)
		owned[name] = observer
	}
	slices.Sort(names)
	return &FencedInventoryVerifier{targets: names, observers: owned}, nil
}

func (v *FencedInventoryVerifier) VerifyInventory(ctx context.Context, input Inventory) (InventoryConfirmation, error) {
	if v == nil {
		return InventoryConfirmation{}, ragy.ErrInvalidArgument
	}
	captured := cloneInventory(input)
	fingerprint, err := captured.Fingerprint()
	if err != nil {
		return InventoryConfirmation{}, err
	}
	names := slices.Clone(captured.Targets)
	slices.Sort(names)
	if !slices.Equal(names, v.targets) {
		return InventoryConfirmation{}, ragy.ErrUnsupported
	}
	var confirmation InventoryConfirmation
	err = v.observe(ctx, captured, 0, func() error {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		confirmation = InventoryConfirmation{
			Namespace:   captured.Namespace,
			Watermark:   captured.Watermark,
			Fingerprint: fingerprint,
			Coverage:    captured.Coverage,
		}
		return nil
	})
	if err != nil {
		return InventoryConfirmation{}, err
	}
	if err = ctx.Err(); err != nil {
		return InventoryConfirmation{}, err
	}
	return confirmation, nil
}

func (v *FencedInventoryVerifier) observe(ctx context.Context, input Inventory, index int, next func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == len(v.targets) {
		return next()
	}
	calls := 0
	var callbackErr error
	err := v.observers[v.targets[index]].ObserveInventory(ctx, cloneInventory(input), func() error {
		calls++
		if calls != 1 {
			return ragy.ErrProtocol
		}
		callbackErr = v.observe(ctx, input, index+1, next)
		return callbackErr
	})
	if err != nil {
		return err
	}
	if calls != 1 {
		return ragy.ErrProtocol
	}
	return callbackErr
}
