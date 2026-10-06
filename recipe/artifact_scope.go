package recipe

import (
	"context"
	"errors"

	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// Artifact rendering protects every standalone failure. Record host callback
// provenance before that wrapper, so only the renderer's own expired timer can
// become a recipe-local bounded stop; a host protection error never can.
func (a *attempt[TIntent, TRequestMeta, TMeta]) artifactOptions(
	options retrieval.ArtifactRenderOptions[TMeta],
) (retrieval.ArtifactRenderOptions[TMeta], func() error) {
	var observed error
	capture := func(err error) error {
		classified := a.callbackError(err)
		observed = errors.Join(observed, classified)
		return classified
	}
	if measure := options.Resource.Measure; measure != nil {
		options.Resource.Measure = func(ctx context.Context, text string) (int64, error) {
			n, err := measure(ctx, text)
			return n, capture(err)
		}
	}
	if clone := options.CloneMeta; clone != nil {
		options.CloneMeta = func(meta TMeta) (TMeta, error) {
			owned, err := clone(meta)
			return owned, capture(err)
		}
	}
	if mapping := options.Mapping; mapping != nil {
		options.Mapping = func(doc retrieval.Document[TMeta]) (source.MappedText, error) {
			mapped, err := mapping(doc)
			return mapped, capture(err)
		}
	}
	if format := options.FormatSnippet; format != nil {
		options.FormatSnippet = func(snippet retrieval.ContextSnippet[TMeta]) (retrieval.FormattedSnippet, error) {
			formatted, err := format(snippet)
			return formatted, capture(err)
		}
	}
	return options, func() error { return observed }
}

func onlyDeadlineCauses(err error) bool {
	if err == nil {
		return false
	}
	if _, independent := errors.AsType[*stageFailureError](err); independent {
		return false
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		if len(causes) == 0 {
			return false
		}
		for _, cause := range causes {
			if !onlyDeadlineCauses(cause) {
				return false
			}
		}
		return true
	}
	if cause := errors.Unwrap(err); cause != nil {
		return onlyDeadlineCauses(cause)
	}
	//nolint:errorlint // Only an exact deadline leaf has timer-only provenance here.
	return err == context.DeadlineExceeded
}
