package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

func TestCompositionR2DegradationSuppressesErrorDiagnostics(t *testing.T) {
	for _, mode := range []string{"protection", "cancellation"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange.
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			input := NewResultSet([]Document[struct{}]{{ID: "secret", Content: "private"}}, nil)
			fusion := &FusionFailureError[struct{}]{
				Cause:        ragy.ErrUnavailable,
				observations: []ResultSet[struct{}]{input},
			}
			m := DegradingMerger[struct{}]{
				Primary: compositionMergerFunc(
					func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
						if mode == "cancellation" {
							cancel()
							return input, fusion
						}
						return input, errors.Join(fusion, access.Protect(ragy.ErrUnavailable))
					},
				),
				Fallback: compositionMergerFunc(
					func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) { panic("fallback") },
				),
			}
			// Act.
			// Act.
			out, err := m.Merge(ctx, input)
			// Assert.
			view, ok := errors.AsType[*FusionFailureError[struct{}]](err)
			// Assert.
			if !out.IsEmpty() || !ok || len(view.Observations()) != 0 {
				t.Fatalf("len=%d diagnostic observations=%d err=%v", out.Len(), len(view.Observations()), err)
			}
		})
	}
}
func TestCompositionR2ProviderCancellationCannotDegrade(t *testing.T) {
	// Arrange.
	calls := 0
	input := NewResultSet([]Document[struct{}]{{ID: "a", Content: "a"}}, nil)
	m := DegradingMerger[struct{}]{
		Primary: compositionMergerFunc(func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
			return input, context.Canceled
		}),
		Fallback: compositionMergerFunc(
			func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) { calls++; return input, nil },
		),
	}
	// Act.
	out, err := m.Merge(t.Context(), input)
	// Assert.
	if !out.IsEmpty() || calls != 0 || !errors.Is(err, context.Canceled) {
		t.Fatalf("len=%d fallback calls=%d err=%v", out.Len(), calls, err)
	}
}

func TestCompositionR2InvalidChainConfigBeforeEarlierProcessor(t *testing.T) {
	// Arrange.
	first := &configProcessor{}
	input := NewResultSet([]Document[struct{}]{{ID: "a", Content: "a"}}, nil)
	// Act.
	_, err := NewPostProcessorChain[struct{}](
		first,
		nil,
	).Process(t.Context(), UnrestrictedRead(), RetrieveOptions{TopK: 1}, input)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) || first.calls != 0 {
		t.Fatalf("earlier processor calls=%d err=%v", first.calls, err)
	}
}
