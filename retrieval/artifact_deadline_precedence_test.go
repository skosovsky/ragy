package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

func TestStandaloneRendererPreservesCancelAndProtocol(t *testing.T) {
	// Arrange.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resource := RuneResource(1000)
	resource.Measure = func(context.Context, string) (int64, error) { cancel(); return 0, ragy.ErrProtocol }
	// Act.
	result, err := (DefaultArtifactRenderer[struct{}]{}).Render(
		ctx,
		UnrestrictedRead(),
		NewResultSet([]Document[struct{}]{{ID: "d", Content: "text"}}, nil),
		ArtifactRenderOptions[struct{}]{Resource: resource, CloneMeta: cloneArtifactValue[struct{}]},
	)
	// Assert.
	if !access.IsProtectionFailure(err) || !errors.Is(err, context.Canceled) || !errors.Is(err, ragy.ErrProtocol) ||
		len(result.Snippets) > 0 {
		t.Fatalf(
			"cause preservation failed: protected=%v canceled=%v protocol=%v err=%v",
			access.IsProtectionFailure(err),
			errors.Is(err, context.Canceled),
			errors.Is(err, ragy.ErrProtocol),
			err,
		)
	}
}

type artifactPrivatePayloadError struct{ Secret string }

func (e *artifactPrivatePayloadError) Error() string { return e.Secret }
func TestStandaloneRendererPreservesTaxonomyWithoutPrivateSidePayload(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		// Arrange.
		ctx, cancel := context.WithCancel(context.Background())
		private := &artifactPrivatePayloadError{Secret: "private-side-payload"}
		resource := RuneResource(1000)
		resource.Measure = func(context.Context, string) (int64, error) {
			if cancelled {
				cancel()
			}
			return 0, errors.Join(access.NonSkippable(context.DeadlineExceeded), ragy.ErrProtocol, private)
		}
		// Act.
		result, err := (DefaultArtifactRenderer[struct{}]{}).Render(
			ctx,
			UnrestrictedRead(),
			NewResultSet([]Document[struct{}]{{ID: "d", Content: "text"}}, nil),
			ArtifactRenderOptions[struct{}]{Resource: resource, CloneMeta: cloneArtifactValue[struct{}]},
		)
		cancel()
		var retained *artifactPrivatePayloadError
		// Assert.
		if !access.IsProtectionFailure(err) || !errors.Is(err, context.DeadlineExceeded) ||
			!errors.Is(err, ragy.ErrProtocol) ||
			errors.As(err, &retained) ||
			len(result.Snippets) > 0 {
			t.Fatalf(
				"cancel=%v protected=%v deadline=%v protocol=%v private=%v err=%v",
				cancelled,
				access.IsProtectionFailure(err),
				errors.Is(err, context.DeadlineExceeded),
				errors.Is(err, ragy.ErrProtocol),
				retained,
				err,
			)
		}
	}
}
